#!/usr/bin/env python3
"""
Analyze memory dumps — find monotonically increasing values
that could be distance, position, or coins.

No need to know exact displayed distance — just find addresses
where value consistently increases between snapshots.
"""
import math

# Known approximate distances (may differ by 2-5m due to timing)
D1_APPROX = 5    # screenshot showed 5m
D2_APPROX = 40   # screenshot showed 40m
D3_APPROX = 64   # screenshot showed 64m


def load_scan(path: str) -> dict[str, float]:
    data = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("0x"):
                try:
                    val = float(parts[1])
                    if not math.isnan(val) and not math.isinf(val):
                        data[parts[0]] = val
                except ValueError:
                    continue
    return data


def main():
    s1 = load_scan("/tmp/ptr_s1.txt")
    s2 = load_scan("/tmp/ptr_d2.txt")
    s3 = load_scan("/tmp/ptr_d3.txt")
    print(f"Loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

    # === Strategy 1: Monotonically increasing, in distance range ===
    print("\n=== MONOTONIC INCREASE (0 < v1 < v2 < v3 < 500) ===")
    mono = []
    for addr in s1:
        v1, v2, v3 = s1.get(addr), s2.get(addr), s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        if not (0 < v1 < v2 < v3 < 500):
            continue
        d12 = v2 - v1
        d23 = v3 - v2
        d13 = v3 - v1
        # Values should span a reasonable range (at least 20)
        if d13 < 20:
            continue
        mono.append((addr, v1, v2, v3, d12, d23))

    # Sort by how close v3 is to ~64 (our expected distance)
    mono.sort(key=lambda x: abs(x[3] - 67))
    print(f"Found {len(mono)} addresses")
    print(f"\nTop 30 (sorted by v3 closest to ~67):")
    for addr, v1, v2, v3, d12, d23 in mono[:30]:
        print(f"  {addr}: {v1:8.2f} → {v2:8.2f} → {v3:8.2f}  "
              f"(d12={d12:.1f}, d23={d23:.1f})")

    # === Strategy 2: Value ≈ distance (with unknown offset) ===
    # displayed = internal - offset
    # D2-D1 = (v2-offset) - (v1-offset) = v2-v1
    # So: v2-v1 should ≈ D2-D1 = 35, v3-v1 should ≈ D3-D1 = 59
    print(f"\n=== DELTA MATCH: d12≈35, d13≈59 (tol=8) ===")
    delta_match = []
    for addr, v1, v2, v3, d12, d23 in mono:
        # The deltas should match displayed deltas
        # But timing makes this ±5m uncertain
        if abs(d12 - 35) < 8 and abs((v3-v1) - 59) < 8:
            offset = v1 - D1_APPROX
            delta_match.append((addr, v1, v2, v3, offset))

    delta_match.sort(key=lambda x: abs(x[4]))  # sort by smallest offset
    print(f"Found {len(delta_match)} addresses")
    for addr, v1, v2, v3, offset in delta_match[:30]:
        print(f"  {addr}: {v1:8.2f} → {v2:8.2f} → {v3:8.2f}  "
              f"(offset={offset:+.1f}, so dist={v1-offset:.0f}→{v2-offset:.0f}→{v3-offset:.0f})")

    # === Strategy 3: Look for fuel (decreasing, 0-1 or 0-100 range) ===
    print(f"\n=== DECREASING (fuel candidate: 0 < v3 < v2 < v1) ===")
    fuel = []
    for addr in s1:
        v1, v2, v3 = s1.get(addr), s2.get(addr), s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        if not (0 < v3 < v2 < v1):
            continue
        if v1 > 1.1 and v1 < 110:  # reasonable fuel range
            fuel.append((addr, v1, v2, v3))

    fuel.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(fuel)} addresses")
    for addr, v1, v2, v3 in fuel[:20]:
        print(f"  {addr}: {v1:8.4f} → {v2:8.4f} → {v3:8.4f}")

    # === Strategy 4: Speed-like (positive during S1-S2, may decrease at S3) ===
    print(f"\n=== SPEED-LIKE (>0 during driving, variable) ===")
    speed = []
    for addr in s1:
        v1, v2, v3 = s1.get(addr), s2.get(addr), s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        # Speed: should be >0 at some point, in range 0-50
        if not (0 < v2 < 50 and 0 < v1 < 50):
            continue
        # Should vary (not constant)
        if abs(v1 - v2) < 0.1 and abs(v2 - v3) < 0.1:
            continue
        # Reasonable speed range
        if max(v1, v2, v3) < 30:
            speed.append((addr, v1, v2, v3))

    print(f"Found {len(speed)} addresses (showing first 10)")
    for addr, v1, v2, v3 in speed[:10]:
        print(f"  {addr}: {v1:8.4f} → {v2:8.4f} → {v3:8.4f}")


if __name__ == "__main__":
    main()
