#!/usr/bin/env python3
"""
Test pipeline: navigate → pvmscan → gas → pvmread
With background game health monitoring and screenshots.
"""
import subprocess
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTAINER = "hcr2-0"
SERIAL = "localhost:5555"
SCREENSHOT_DIR = "/tmp/pvm_test"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def get_pid():
    """Get game PID inside container, or None if not running."""
    try:
        r = subprocess.run(
            ["docker", "exec", CONTAINER, "pidof", "com.fingersoft.hcr2"],
            capture_output=True, text=True, timeout=3
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def screenshot(tag=""):
    """Take screenshot and save with timestamp."""
    try:
        from hillclimb.capture import ScreenCapture
        from hillclimb.vision import VisionAnalyzer
        import cv2
        cap = ScreenCapture(SERIAL)
        frame = cap.grab()
        if frame is not None:
            va = VisionAnalyzer()
            result = va.analyze(frame)
            ts = time.strftime("%H%M%S")
            fname = f"{SCREENSHOT_DIR}/{ts}_{tag}.png"
            cv2.imwrite(fname, frame)
            return result.game_state, fname
    except Exception as e:
        print(f"  [SCREENSHOT ERROR] {e}")
    return None, None


# ── Background health monitor ──
_stop_monitor = threading.Event()
_game_died_at = None


def health_monitor(pid, interval=0.5):
    """Check if game process is alive every interval seconds."""
    global _game_died_at
    t0 = time.time()
    while not _stop_monitor.is_set():
        current_pid = get_pid()
        elapsed = time.time() - t0
        if current_pid is None or current_pid != pid:
            _game_died_at = elapsed
            print(f"\n  *** GAME DIED at t={elapsed:.1f}s! (was pid={pid}) ***\n")
            break
        time.sleep(interval)


def main():
    # ── Step 0: Check game is running ──
    pid = get_pid()
    if not pid:
        print("Game not running, starting...")
        subprocess.run(["docker", "exec", CONTAINER, "am", "start", "-n",
                        "com.fingersoft.hcr2/.AppActivity"], timeout=5)
        time.sleep(5)
        pid = get_pid()
        if not pid:
            print("FAILED to start game")
            return

    print(f"Game PID: {pid}")
    state, fname = screenshot("initial")
    print(f"Initial state: {state} ({fname})")

    # ── Step 1: Navigate to RACING ──
    print("\n=== Step 1: Navigate to RACING ===")
    from hillclimb.capture import ScreenCapture
    from hillclimb.vision import VisionAnalyzer
    from hillclimb.controller import ADBController
    from hillclimb.navigator import Navigator

    cap = ScreenCapture(SERIAL)
    va = VisionAnalyzer()
    ctrl = ADBController(SERIAL)
    nav = Navigator(ctrl, cap, va, env_index=0)

    t_nav_start = time.time()
    success = nav.ensure_racing(timeout=90)
    t_nav = time.time() - t_nav_start
    print(f"Nav result: {success} in {t_nav:.1f}s")

    if not success:
        state, fname = screenshot("nav_failed")
        print(f"Nav failed. State: {state} ({fname})")
        return

    # Re-check PID (may have changed after relaunch)
    pid = get_pid()
    if not pid:
        print("Game died during navigation!")
        return
    print(f"Game PID after nav: {pid}")

    state, fname = screenshot("racing_start")
    print(f"State: {state} ({fname})")

    # ── Start health monitor ──
    _stop_monitor.clear()
    monitor = threading.Thread(target=health_monitor, args=(pid, 0.3), daemon=True)
    monitor.start()
    t0 = time.time()

    # ── Step 2: pvmscan ──
    print(f"\n=== Step 2: pvmscan (t={time.time()-t0:.1f}s) ===")
    scan_result = subprocess.run(
        ["docker", "exec", CONTAINER, "/data/local/tmp/pvmscan", pid],
        capture_output=True, text=True, timeout=10
    )
    t_scan = time.time() - t0
    print(f"pvmscan took {t_scan:.2f}s")
    print(f"stderr: {scan_result.stderr.strip()}")
    print(f"stdout: {scan_result.stdout.strip()}")

    if scan_result.returncode != 0:
        print("pvmscan failed!")
        _stop_monitor.set()
        return

    # Check if game survived scan
    time.sleep(0.2)
    if _game_died_at is not None:
        print(f"Game died at t={_game_died_at:.1f}s (during/after scan)")
        screenshot("died_after_scan")
        _stop_monitor.set()
        return

    body_addr = scan_result.stdout.strip().split('\n')[-1]
    addr_int = int(body_addr, 16)
    # Read 128 bytes starting from pos_y (body_x - 4)
    read_addr_hex = format(addr_int - 4, 'x')
    print(f"Body: {body_addr}, read from: 0x{read_addr_hex}")

    # ── Step 3: Wait a bit, then screenshot ──
    print(f"\n=== Step 3: Post-scan check (t={time.time()-t0:.1f}s) ===")
    time.sleep(1.0)
    if _game_died_at is not None:
        print(f"Game died at t={_game_died_at:.1f}s (1s after scan)")
        screenshot("died_1s_after_scan")
        _stop_monitor.set()
        return
    state, fname = screenshot("post_scan_1s")
    print(f"1s after scan: state={state}, game alive ({fname})")

    time.sleep(1.0)
    if _game_died_at is not None:
        print(f"Game died at t={_game_died_at:.1f}s (2s after scan)")
        _stop_monitor.set()
        return
    print(f"2s after scan: game alive (t={time.time()-t0:.1f}s)")

    # ── Step 4: Start gas ──
    print(f"\n=== Step 4: Gas + pvmread (t={time.time()-t0:.1f}s) ===")
    gas_proc = subprocess.Popen(
        ["docker", "exec", CONTAINER, "input", "swipe", "720", "430", "720", "430", "12000"]
    )
    time.sleep(0.5)

    if _game_died_at is not None:
        print(f"Game died at t={_game_died_at:.1f}s (after gas start)")
        _stop_monitor.set()
        return

    # ── Step 5: pvmread ──
    print(f"Starting pvmread (t={time.time()-t0:.1f}s)")
    read_result = subprocess.run(
        ["docker", "exec", CONTAINER, "/data/local/tmp/pvmread",
         pid, read_addr_hex, "128", "200", "40"],
        capture_output=True, text=True, timeout=15
    )
    t_read_end = time.time() - t0
    print(f"pvmread finished at t={t_read_end:.1f}s")
    print(f"stderr: {read_result.stderr.strip()}")

    # Parse and display results
    lines = read_result.stdout.strip().split('\n')
    print(f"\n=== Results: {len(lines)} reads ===")
    print(f"{'time':>8s} {'pos_x':>10s} {'pos_y':>10s} {'vel_x':>10s} {'vel_y':>10s}")
    prev_x = None
    for line in lines:
        parts = line.split()
        if len(parts) < 17:
            continue
        t = parts[0]  # t=X.XXX
        pos_y = float(parts[1])
        pos_x = float(parts[2])
        # vel_x at offset +60 from body_x = offset +64 from pos_y = float index 16
        # vel_y at offset +64 from body_x = offset +68 from pos_y = float index 17
        vel_x = float(parts[16]) if len(parts) > 16 else 0
        vel_y = float(parts[17]) if len(parts) > 17 else 0
        changed = "" if prev_x is None else (" <<<" if pos_x != prev_x else "")
        print(f"{t:>8s} {pos_x:>10.2f} {pos_y:>10.2f} {vel_x:>10.3f} {vel_y:>10.3f}{changed}")
        prev_x = pos_x

    gas_proc.wait()

    # ── Final screenshot ──
    state, fname = screenshot("final")
    print(f"\nFinal: state={state}, game_died_at={_game_died_at} ({fname})")

    _stop_monitor.set()


if __name__ == "__main__":
    main()
