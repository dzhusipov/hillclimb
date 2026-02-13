#!/usr/bin/env python3
"""
Automated memory scanning for HCR2 distance value.
Strategy:
1. Navigate to race, drive, wait for RESULTS
2. Read distance from RESULTS screen
3. Scan memory for that float value
4. Start new race, drive different distance, wait for RESULTS
5. Rescan candidates → narrow down to distance address

Uses docker exec for memory tools, adb for game control.
"""
import subprocess
import sys
import time
import re


CONTAINER = "hcr2-0"
ADB_SERIAL = "localhost:5555"


def adb(cmd: str, timeout: int = 5):
    """Run ADB command."""
    subprocess.run(
        ["adb", "-s", ADB_SERIAL, "shell", cmd],
        capture_output=True, timeout=timeout,
    )


def tap(x: int, y: int, delay: float = 0.5):
    adb(f"input tap {x} {y}")
    time.sleep(delay)


def swipe(x: int, y: int, dur_ms: int = 200, delay: float = 0.5):
    adb(f"input swipe {x} {y} {x} {y} {dur_ms}")
    time.sleep(delay)


def screenshot(path: str):
    with open(path, "wb") as f:
        subprocess.run(
            ["adb", "-s", ADB_SERIAL, "shell", "screencap", "-p"],
            stdout=f, stderr=subprocess.DEVNULL, timeout=10,
        )


def get_pid() -> int:
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "pidof", "com.fingersoft.hcr2"],
        capture_output=True, timeout=5,
    )
    out = r.stdout.decode().strip()
    return int(out) if out else 0


def memscan(pid: int, value: float, tol: float = 1.0) -> str:
    """Run memscan, return output."""
    r = subprocess.run(
        ["docker", "exec", CONTAINER,
         "/data/local/tmp/memscan", str(pid), str(value), str(tol)],
        capture_output=True, timeout=60,
    )
    stderr = r.stderr.decode().strip()
    print(f"  memscan: {stderr}")
    return r.stdout.decode()


def memdump(pid: int, addrs_text: str) -> str:
    """Run memdump with addresses on stdin."""
    r = subprocess.run(
        ["docker", "exec", "-i", CONTAINER,
         "/data/local/tmp/memdump", str(pid)],
        input=addrs_text.encode(),
        capture_output=True, timeout=30,
    )
    stderr = r.stderr.decode().strip()
    print(f"  memdump: {stderr}")
    return r.stdout.decode()


def launch_game():
    """Launch HCR2 and navigate to VEHICLE_SELECT."""
    print("Launching HCR2...")
    subprocess.run(
        ["adb", "-s", ADB_SERIAL, "shell",
         "am", "start", "-n", "com.fingersoft.hcr2/.AppActivity"],
        capture_output=True, timeout=5,
    )
    time.sleep(8)  # loading screen
    # Dismiss OFFLINE + navigate
    tap(155, 25, delay=1.0)   # ADVENTURE tab
    tap(650, 295, delay=3.0)  # RACE button
    print("At VEHICLE_SELECT")


def start_race_and_drive(gas_ms: int = 200, coast_wait: float = 5.0):
    """Start race and drive briefly."""
    tap(730, 445, delay=4.5)  # START (wait for load + flag animation)
    # Skip DOUBLE_COINS if it appears
    tap(400, 300, delay=0.5)  # skip button area
    # Short gas
    swipe(750, 430, dur_ms=gas_ms, delay=coast_wait)


def wait_for_results(max_wait: int = 60) -> bool:
    """Wait for RESULTS screen (or crash). Returns True if RESULTS reached."""
    for i in range(max_wait):
        time.sleep(1)
        path = f"/tmp/hcr2_check_{i}.png"
        screenshot(path)
        # Quick check: read file size (RESULTS has specific pattern)
        # Just check if game is still running
        pid = get_pid()
        if pid == 0:
            print("  Game crashed!")
            return False
        # Check for RESULTS: read a portion of the image
        # For simplicity, check if RETRY button area has green
        r = subprocess.run(
            ["docker", "exec", CONTAINER,
             "sh", "-c", f"cat /proc/{pid}/status | head -1"],
            capture_output=True, timeout=5,
        )
        if r.returncode != 0:
            print("  Process gone!")
            return False
    return True


def race_to_completion(gas_ms: int = 300):
    """Start race, press gas briefly, wait for fuel to run out."""
    print(f"Starting race (gas={gas_ms}ms)...")
    tap(730, 445, delay=5.0)  # START
    # Short gas press
    swipe(750, 430, dur_ms=gas_ms, delay=0)
    print("  Gas pressed, waiting for race to end...")
    # Wait for the car to crash/run out of fuel
    # Poll the screen looking for RESULTS indicators
    time.sleep(15)  # typical short race
    # After driver_down → touch to continue → results
    # Let the game auto-progress (or help it)
    tap(400, 240, delay=1.0)  # tap center (TOUCH_TO_CONTINUE)
    tap(400, 240, delay=1.0)  # tap again (just in case)
    time.sleep(2)


def main():
    # Step 0: Launch game
    pid = get_pid()
    if pid == 0:
        launch_game()
        pid = get_pid()
    else:
        # Game already running, navigate to vehicle select
        tap(155, 25, delay=1.0)   # ADVENTURE
        tap(650, 295, delay=3.0)  # RACE

    print(f"PID: {pid}")
    if pid == 0:
        print("ERROR: HCR2 not running!")
        return

    results = []  # list of (distance_label, scan_data)

    # RACE 1: very short drive
    print("\n=== RACE 1 ===")
    race_to_completion(gas_ms=200)

    pid = get_pid()
    if pid == 0:
        print("Game crashed after race 1!")
        return

    # Take screenshot to see distance
    screenshot("/tmp/hcr2_race1_result.png")
    print("  Race 1 results screenshot saved")

    # Scan for values 5-300 (we don't know exact distance yet)
    # Better: ask user to read distance from screenshot
    # For automation: scan for common short distances
    # Let's just dump a broad scan
    print("  Scanning (broad: 10-200, tol=200)...")
    scan1 = memscan(pid, 50.0, 200.0)
    lines = scan1.strip().split("\n") if scan1.strip() else []
    print(f"  Scan 1: {len(lines)} candidates")

    if not lines:
        print("No matches found!")
        return

    # Extract addresses
    addrs_text = "\n".join(line.split()[0] for line in lines if line.strip())

    # Save scan 1 data
    with open("/tmp/memscan_race1.txt", "w") as f:
        f.write(scan1)
    print("  Saved to /tmp/memscan_race1.txt")

    # RACE 2: slightly longer drive
    print("\n=== RACE 2 (RETRY) ===")
    tap(87, 430, delay=3.0)  # RETRY
    race_to_completion(gas_ms=800)

    pid2 = get_pid()
    if pid2 == 0:
        print("Game crashed after race 2!")
        # Still analyze what we have
    elif pid2 != pid:
        print(f"PID changed: {pid} -> {pid2}! Addresses invalid.")
        return
    else:
        screenshot("/tmp/hcr2_race2_result.png")
        print("  Race 2 results screenshot saved")

        # Dump ALL candidate addresses with current values
        print("  Dumping all candidates...")
        dump2 = memdump(pid, addrs_text)
        with open("/tmp/memscan_race2_dump.txt", "w") as f:
            f.write(dump2)
        print(f"  Saved to /tmp/memscan_race2_dump.txt")

    # RACE 3: even longer drive
    print("\n=== RACE 3 (RETRY) ===")
    pid = get_pid()
    if pid == 0:
        print("Game not running, stopping.")
    else:
        tap(87, 430, delay=3.0)  # RETRY
        race_to_completion(gas_ms=2000)

        pid3 = get_pid()
        if pid3 and pid3 == pid:
            screenshot("/tmp/hcr2_race3_result.png")
            print("  Race 3 results screenshot saved")

            dump3 = memdump(pid, addrs_text)
            with open("/tmp/memscan_race3_dump.txt", "w") as f:
                f.write(dump3)
            print(f"  Saved to /tmp/memscan_race3_dump.txt")

    print("\n=== DONE ===")
    print("Analyze results:")
    print("  Race 1 scan: /tmp/memscan_race1.txt")
    print("  Race 2 dump: /tmp/memscan_race2_dump.txt")
    print("  Race 3 dump: /tmp/memscan_race3_dump.txt")
    print("  Screenshots: /tmp/hcr2_race{1,2,3}_result.png")
    print("Look for addresses where values match displayed distances!")


if __name__ == "__main__":
    main()
