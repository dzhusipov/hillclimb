#!/usr/bin/env python3
"""
Test which operation triggers HCR2 anti-cheat.
Modes:
  0: only read /proc/PID/maps
  1: one process_vm_readv (4 bytes)
  2: 10x process_vm_readv (4 bytes, 100ms apart)
  3: scan ~10MB with process_vm_readv
  4: scan full heap (~600MB) with process_vm_readv
"""
import subprocess
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTAINER = "hcr2-0"
SERIAL = "localhost:5555"
MONITOR_SECONDS = 15


def get_pid():
    try:
        r = subprocess.run(["docker", "exec", CONTAINER, "pidof", "com.fingersoft.hcr2"],
                          capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except:
        return None


def navigate_to_racing():
    from hillclimb.capture import ScreenCapture
    from hillclimb.vision import VisionAnalyzer
    from hillclimb.controller import ADBController
    from hillclimb.navigator import Navigator

    cap = ScreenCapture(SERIAL)
    va = VisionAnalyzer()
    ctrl = ADBController(SERIAL)
    nav = Navigator(ctrl, cap, va, env_index=0)
    return nav.ensure_racing(timeout=90)


def ensure_game_running():
    pid = get_pid()
    if not pid:
        subprocess.run(["docker", "exec", CONTAINER, "am", "start", "-n",
                        "com.fingersoft.hcr2/.AppActivity"], timeout=5)
        time.sleep(5)
        pid = get_pid()
    return pid


def run_test(mode):
    print(f"\n{'='*50}")
    print(f"=== MODE {mode} ===")
    print(f"{'='*50}")

    # Ensure game running
    pid = ensure_game_running()
    if not pid:
        print("  FAILED: game not running")
        return None

    # Navigate to RACING
    print("  Navigating to RACING...")
    if not navigate_to_racing():
        print("  FAILED: navigation")
        return None

    pid = get_pid()
    if not pid:
        print("  FAILED: game died during nav")
        return None
    print(f"  RACING, PID={pid}")

    # Start gas
    gas = subprocess.Popen(
        ["docker", "exec", CONTAINER, "input", "swipe", "720", "430", "720", "430", "20000"]
    )
    time.sleep(0.5)

    # Run pvmtest with this mode
    print(f"  Running pvmtest mode {mode}...")
    t0 = time.time()
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "/data/local/tmp/pvmtest", pid, str(mode)],
        capture_output=True, text=True, timeout=30
    )
    test_time = time.time() - t0
    print(f"  pvmtest took {test_time:.2f}s: {r.stderr.strip()}")

    # Monitor for MONITOR_SECONDS
    print(f"  Monitoring game for {MONITOR_SECONDS}s...")
    t_start = time.time()
    died_at = None
    while time.time() - t_start < MONITOR_SECONDS:
        p = get_pid()
        if p is None or p != pid:
            died_at = time.time() - t_start
            break
        time.sleep(0.3)

    gas.wait()

    if died_at is not None:
        print(f"  RESULT: GAME DIED at +{died_at:.1f}s after pvmtest")
        return died_at
    else:
        print(f"  RESULT: GAME SURVIVED {MONITOR_SECONDS}s!")
        # Kill the race for next test
        subprocess.run(["docker", "exec", CONTAINER, "am", "force-stop", "com.fingersoft.hcr2"], timeout=5)
        time.sleep(1)
        return -1  # survived


def main():
    modes = [0, 1, 2, 3, 4]
    if len(sys.argv) > 1:
        modes = [int(x) for x in sys.argv[1:]]

    results = {}
    for mode in modes:
        result = run_test(mode)
        results[mode] = result
        time.sleep(2)

    print(f"\n{'='*50}")
    print("=== SUMMARY ===")
    print(f"{'='*50}")
    for mode, result in results.items():
        if result is None:
            status = "FAILED (test setup)"
        elif result < 0:
            status = "SURVIVED"
        else:
            status = f"DIED at +{result:.1f}s"
        descriptions = {
            0: "maps read only",
            1: "1x process_vm_readv (4B)",
            2: "10x process_vm_readv (4B, 100ms)",
            3: "scan ~10MB",
            4: "scan full heap ~600MB",
        }
        desc = descriptions.get(mode, f"mode {mode}")
        print(f"  Mode {mode} ({desc}): {status}")


if __name__ == "__main__":
    main()
