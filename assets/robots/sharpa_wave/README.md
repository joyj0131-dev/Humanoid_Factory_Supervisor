# Sharpa Wave hand assets (Phase 4, 35th session)

Phase 4 Grasp Track's new target end-effector (replacing Dex3-1 as the
active development target; Dex3-1 is preserved as the legacy baseline).

## Upstream

- Repository: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
- Pinned commit: `6eea427eb24189519f32b9f21674cd534d3f973c` ("Add license and notice")
- License: Apache License 2.0 -- see `LICENSE.txt` (verbatim) and `NOTICE.txt`
  (verbatim, credits MuJoCo/DeepMind and box-collision code by Svetoslav Kolev)
- Upstream README describing the model family: `UPSTREAM_README.md` (verbatim copy)

## What is and is not vendored into this git repo

The upstream repo is ~190MB total (mostly Isaac/USD-only configuration
files this project never uses -- `*.usd`, `*.usda`, per-variant
`configuration/` subfolders, `config.yaml`, ROS2 launch/RViz files).
Only the MuJoCo-relevant subset is used here: `{left,right}_sharpa_wave.xml`,
the `_with_wrist` and `_with_flange` XML variants, the reference URDF, and
the visual/collision STL meshes (~23MB total).

That ~23MB subset is deliberately **not** committed to this repo's git
history (large binary meshes are a hard-to-reverse addition to history).
Instead:
- `checksums.sha256` (tracked) lists every vendored file's exact sha256,
  pinned to the commit above -- this is the source of truth for exactly
  which files belong to the vendored subset.
- `scripts/install_sharpa_wave_assets.py` (tracked, repo root `scripts/`)
  re-downloads that exact file set from the pinned commit and verifies
  every file against `checksums.sha256`. Run it after cloning:
  ```
  python3 scripts/install_sharpa_wave_assets.py
  ```
- `left_sharpa_wave/` and `right_sharpa_wave/` (the actual XML+mesh
  content) are gitignored; verified reproducible bit-for-bit from a
  clean state during the 35th session (re-downloaded copies were
  byte-identical to the originally vendored ones via `diff -rq`).

## Model facts (verified directly from the XML, 35th session audit)

- 22 active joints per hand: thumb (CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP =
  5), index/middle/ring (MCP_FE, MCP_AA, PIP, DIP = 4 each), pinky (CMC,
  MCP_FE, MCP_AA, PIP, DIP = 5). 5+4+4+4+5 = 22, matching the vendor's
  "22 active DoF" claim exactly.
- 22 `<position>` actuators per hand, one per joint (kp/dampratio tuned
  per joint class, `inheritrange="1.0"`).
- Left/right are exact mirrors (identical joint names with `left_`/
  `right_` prefixes, confirmed by diff).
- `_with_wrist` variants add a RIGID (no extra DoF) wrist standoff geom
  (`wrist_B.STL` visual + `wrist_collision.STL` collision) on the SAME
  root body, shifting the palm's own geometry +29mm along local Z --
  this represents physical wrist-adapter thickness, not an articulated
  wrist joint. `_with_flange` variants omit this standoff for direct
  flange mounting.
- **No `<sensor>` elements exist in any of these MJCF files** (base,
  `_with_wrist`, or `_with_flange`, left and right all confirmed via
  direct grep). The real-hardware "240x240 tactile array" / slip
  inference is NOT present in the MuJoCo model -- do not assume it or
  fabricate synthetic dense tactile values. What IS present: fingertip
  **elastomer collision geoms** (`*_elastomer`, `thumb_elastomer`) with
  a softened `solref="0.06 0.9"` for a more compliant contact response,
  and a `<contact><exclude>` list covering each finger's own adjacent
  links plus palm-to-proximal-link pairs (NOT between different
  fingers, so real inter-finger self-collision is still detected).
  Any per-finger contact/force observation this project uses must be
  derived from MuJoCo's own `mj_contactForce`/`data.contact`, explicitly
  labeled as a **contact-force-based observation**, never as a
  simulated hardware tactile array.
