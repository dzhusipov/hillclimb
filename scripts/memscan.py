#!/usr/bin/env python3
"""Memory scanner for HCR2 — find float values in game process memory."""

import struct
import subprocess
import sys
import time


def get_pid(container: str) -> int:
    r = subprocess.run(
        ["docker", "exec", container, "pidof", "com.fingersoft.hcr2"],
        capture_output=True, timeout=5,
    )
    return int(r.stdout.decode().strip())


def get_heap_regions(container: str, pid: int) -> list[tuple[int, int]]:
    """Get writable heap regions (scudo allocator + anon)."""
    r = subprocess.run(
        ["docker", "exec", container, "cat", f"/proc/{pid}/maps"],
        capture_output=True, timeout=5,
    )
    regions = []
    for line in r.stdout.decode().splitlines():
        parts = line.split()
        perms = parts[1]
        if "w" not in perms:
            continue
        start, end = (int(x, 16) for x in parts[0].split("-"))
        size = end - start
        if size > 50_000_000 or size < 4096:
            continue
        name = parts[-1] if len(parts) > 5 else ""
        if "/dev/" in name or "/system/" in name:
            continue
        regions.append((start, end))
    return regions


def read_mem(container: str, pid: int, start: int, size: int) -> bytes:
    """Read memory chunk via dd on /proc/PID/mem."""
    cmd = (
        f"dd if=/proc/{pid}/mem bs=1 skip={start} count={size} 2>/dev/null"
    )
    r = subprocess.run(
        ["docker", "exec", container, "sh", "-c", cmd],
        capture_output=True, timeout=10,
    )
    return r.stdout


def scan_float(container: str, pid: int, target: float, tolerance: float = 0.5) -> list[int]:
    """Scan all heap for float value within tolerance."""
    regions = get_heap_regions(container, pid)
    target_bytes = struct.pack("<f", target)
    matches = []
    total_scanned = 0

    for start, end in regions:
        size = end - start
        if size > 10_000_000:  # skip very large for speed
            continue
        try:
            data = read_mem(container, pid, start, size)
        except Exception:
            continue
        if len(data) < 4:
            continue
        total_scanned += len(data)

        # Scan for float matches
        for offset in range(0, len(data) - 3, 4):
            val = struct.unpack_from("<f", data, offset)[0]
            if abs(val - target) < tolerance:
                addr = start + offset
                matches.append((addr, val))

    print(f"Scanned {total_scanned / 1024 / 1024:.1f} MB, found {len(matches)} matches")
    return matches


def rescan(container: str, pid: int, addrs: list[int], target: float, tolerance: float = 0.5) -> list[int]:
    """Re-read specific addresses and filter by new target value."""
    remaining = []
    for addr, _ in addrs:
        try:
            data = read_mem(container, pid, addr, 4)
            if len(data) == 4:
                val = struct.unpack("<f", data)[0]
                if abs(val - target) < tolerance:
                    remaining.append((addr, val))
        except Exception:
            continue
    print(f"Rescan: {len(remaining)}/{len(addrs)} remain for target={target}")
    return remaining


if __name__ == "__main__":
    container = sys.argv[1] if len(sys.argv) > 1 else "hcr2-0"
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 181.0

    pid = get_pid(container)
    print(f"PID: {pid}, scanning for float {target}...")

    t0 = time.time()
    matches = scan_float(container, pid, target, tolerance=1.0)
    print(f"Scan took {time.time() - t0:.1f}s")

    if matches:
        print(f"\nFirst 20 matches:")
        for addr, val in matches[:20]:
            print(f"  0x{addr:016x} = {val:.2f}")
