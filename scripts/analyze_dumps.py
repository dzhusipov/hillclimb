#!/usr/bin/env python3
"""
Analyze memory dumps from 3 race snapshots.
Find addresses where values correlate with displayed distances.
"""

# Displayed distances at each snapshot
D1 = 5.0    # ptr_s1.txt (initial scan)
D2 = 40.0   # ptr_d2.txt
D3 = 64.0   # ptr_d3.txt

TOLERANCE = 3.0  # how close float must be to displayed distance

def load_scan(path: str) -> dict[str, float]:
    """Load address -> value mapping."""
    data = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("0x"):
                try:
                    addr = parts[0]
                    val = float(parts[1])
                    data[addr] = val
                except ValueError:
                    continue
    return data


def main():
    print("Loading dumps...")
    s1 = load_scan("/tmp/ptr_s1.txt")
    s2 = load_scan("/tmp/ptr_d2.txt")
    s3 = load_scan("/tmp/ptr_d3.txt")

    print(f"S1: {len(s1)} addrs (distance={D1}m)")
    print(f"S2: {len(s2)} addrs (distance={D2}m)")
    print(f"S3: {len(s3)} addrs (distance={D3}m)")

    # Strategy 1: Direct match — value ≈ displayed distance at all 3 points
    print(f"\n=== Direct match (tol={TOLERANCE}) ===")
    direct_matches = []
    for addr in s1:
        v1 = s1.get(addr)
        v2 = s2.get(addr)
        v3 = s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        if (abs(v1 - D1) < TOLERANCE and
            abs(v2 - D2) < TOLERANCE and
            abs(v3 - D3) < TOLERANCE):
            direct_matches.append((addr, v1, v2, v3))

    print(f"Found {len(direct_matches)} addresses matching all 3 distances:")
    for addr, v1, v2, v3 in direct_matches[:50]:
        print(f"  {addr}: {v1:8.2f} → {v2:8.2f} → {v3:8.2f}")

    # Strategy 2: Proportional match — value changes proportionally to distance
    # ratio = (v2-v1)/(D2-D1) ≈ (v3-v1)/(D3-D1)
    print(f"\n=== Proportional match ===")
    prop_matches = []
    for addr in s1:
        v1 = s1.get(addr)
        v2 = s2.get(addr)
        v3 = s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        dv21 = v2 - v1
        dv31 = v3 - v1
        dd21 = D2 - D1  # 23
        dd31 = D3 - D1  # 38

        if dd21 == 0 or dd31 == 0:
            continue

        ratio21 = dv21 / dd21
        ratio31 = dv31 / dd31

        # Ratios should be similar and positive
        if ratio21 <= 0 or ratio31 <= 0:
            continue
        if abs(ratio21 - ratio31) / max(ratio21, ratio31) < 0.15:
            prop_matches.append((addr, v1, v2, v3, ratio21))

    prop_matches.sort(key=lambda x: abs(x[4] - 1.0))  # sort by how close ratio is to 1.0
    print(f"Found {len(prop_matches)} proportional matches:")
    for addr, v1, v2, v3, ratio in prop_matches[:50]:
        print(f"  {addr}: {v1:10.2f} → {v2:10.2f} → {v3:10.2f}  (ratio={ratio:.4f})")

    # Strategy 3: Value ≈ distance * some_constant
    print(f"\n=== Scaled match (value = distance * K) ===")
    scale_matches = []
    for addr in s1:
        v1 = s1.get(addr)
        v2 = s2.get(addr)
        v3 = s3.get(addr)
        if v1 is None or v2 is None or v3 is None:
            continue
        if D1 == 0 or v1 == 0:
            continue
        k1 = v1 / D1
        k2 = v2 / D2 if D2 != 0 else 0
        k3 = v3 / D3 if D3 != 0 else 0
        if k1 <= 0 or k2 <= 0 or k3 <= 0:
            continue
        # All K values should be similar
        avg_k = (k1 + k2 + k3) / 3
        if avg_k == 0:
            continue
        spread = max(abs(k1-avg_k), abs(k2-avg_k), abs(k3-avg_k)) / avg_k
        if spread < 0.1:  # 10% spread
            scale_matches.append((addr, v1, v2, v3, avg_k))

    scale_matches.sort(key=lambda x: abs(x[4] - 1.0))
    print(f"Found {len(scale_matches)} scaled matches:")
    for addr, v1, v2, v3, k in scale_matches[:50]:
        print(f"  {addr}: {v1:10.2f} → {v2:10.2f} → {v3:10.2f}  (K={k:.4f})")


if __name__ == "__main__":
    main()
