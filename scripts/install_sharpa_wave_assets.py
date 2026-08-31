"""Installs the Sharpa Wave hand MuJoCo assets (Phase 4, 35th session).

The upstream repository (sharpa-robotics/sharpa-urdf-usd-xml) is ~190MB
total (mostly Isaac/USD-only configuration files this project never
uses), so the MJCF-relevant subset (left/right hand XML + collision/
visual meshes, ~23MB) is NOT committed to this repo's git history.
Instead this script re-downloads that exact subset, pinned to the
upstream commit below, and verifies every file against the checksum
manifest also committed here (assets/robots/sharpa_wave/checksums.sha256)
-- so a fresh clone of THIS project can always reproduce the Sharpa
assets bit-for-bit, without needing git access to the (much larger)
upstream repo.

Run with:
    python3 scripts/install_sharpa_wave_assets.py

Upstream: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
Pinned commit: 6eea427eb24189519f32b9f21674cd534d3f973c ("Add license and notice")
License: Apache License 2.0 (see assets/robots/sharpa_wave/LICENSE.txt)
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "robots" / "sharpa_wave"
CHECKSUM_FILE = ASSET_ROOT / "checksums.sha256"

UPSTREAM_REPO = "sharpa-robotics/sharpa-urdf-usd-xml"
UPSTREAM_COMMIT = "6eea427eb24189519f32b9f21674cd534d3f973c"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/wave_01"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_manifest() -> list[tuple[str, str]]:
    """Returns [(relative_path, expected_sha256), ...] from the committed
    checksum manifest -- this manifest, not this script, is the source
    of truth for exactly which files belong to the vendored subset."""
    entries = []
    for line in CHECKSUM_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel_path = line.split(None, 1)
        entries.append((rel_path.strip(), digest.strip()))
    return entries


def main() -> int:
    if not CHECKSUM_FILE.exists():
        print(f"ERROR: checksum manifest not found at {CHECKSUM_FILE}", file=sys.stderr)
        return 1

    entries = parse_manifest()
    print(f"Installing {len(entries)} Sharpa Wave asset files from {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:12]}...")

    n_ok, n_downloaded, n_failed = 0, 0, 0
    for rel_path, expected_sha in entries:
        dest = ASSET_ROOT / rel_path
        if dest.exists() and sha256_of(dest) == expected_sha:
            n_ok += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW_BASE}/{rel_path}"
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to download {rel_path}: {e}", file=sys.stderr)
            n_failed += 1
            continue
        actual_sha = sha256_of(dest)
        if actual_sha != expected_sha:
            print(f"CHECKSUM MISMATCH for {rel_path}: expected {expected_sha}, got {actual_sha}", file=sys.stderr)
            n_failed += 1
            continue
        n_downloaded += 1

    print(f"\n{n_ok} already present and verified, {n_downloaded} downloaded and verified, {n_failed} failed.")
    if n_failed:
        print("Installation INCOMPLETE -- do not proceed to Sharpa integration until this is fixed.", file=sys.stderr)
        return 1
    print("Installation complete and verified against the committed checksum manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
