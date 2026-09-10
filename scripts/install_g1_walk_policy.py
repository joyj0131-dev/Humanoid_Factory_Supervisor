#!/usr/bin/env python3
"""Fetch Unitree's pre-trained G1 locomotion policy (BSD 3-Clause).

The binary is deliberately not committed, matching how the Sharpa Wave meshes
are handled: this script reproduces it and verifies the recorded checksum.

    python3 scripts/install_g1_walk_policy.py

See assets/policies/g1_walk/NOTICE for provenance and license.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "assets" / "policies" / "g1_walk"
TARGET = TARGET_DIR / "motion.pt"
URL = (
    "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/"
    "deploy/pre_train/g1/motion.pt"
)


def expected_sha256() -> str:
    line = (TARGET_DIR / "checksums.sha256").read_text().strip().split()
    return line[0]


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    want = expected_sha256()

    if TARGET.exists():
        have = hashlib.sha256(TARGET.read_bytes()).hexdigest()
        if have == want:
            print(f"already installed and verified: {TARGET}")
            return 0
        print(f"checksum mismatch at {TARGET}; re-downloading", file=sys.stderr)

    print(f"downloading {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read()
    except Exception as exc:  # noqa: BLE001 - report the real reason and stop
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    have = hashlib.sha256(data).hexdigest()
    if have != want:
        print(f"checksum mismatch: expected {want}, got {have}", file=sys.stderr)
        print("Refusing to install. Upstream may have republished the policy.", file=sys.stderr)
        return 1

    TARGET.write_bytes(data)
    print(f"installed {TARGET} ({len(data)} bytes, sha256 verified)")
    print("License: BSD 3-Clause, Unitree Robotics. See assets/policies/g1_walk/NOTICE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
