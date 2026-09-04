"""Sharpa Wave hand joint naming and four-group grasp synergy.

Naming: model_builder.attach_sharpa_hands() attaches each side's Sharpa
spec with prefix=f"{side}_", and the source XML's own bodies/joints are
ALREADY side-prefixed (e.g. "left_thumb_CMC_FE") -- so every resulting
name in the compiled model is "{side}_{side}_..." (e.g.
"left_left_thumb_CMC_FE"). This is a cosmetic redundancy from the
vendored XML's own convention, not a bug (see model_builder.py's
attach_sharpa_hands docstring); sharpa_joint()/sharpa_actuator() below
encode it in ONE place so no other file has to know about it.

Per-finger joint role split (curl vs preshape/spread), reasoned from real
anthropomorphic hand kinematics since no hardware ground truth is
available for this preshape choice (disclosed, not measured -- see
FIVE_FINGER_PRESHAPE targets below):
  - "curl" joints flex/extend the finger to open/close on an object --
    driven by that finger GROUP's own open<->close synergy scalar.
  - "preshape" (abduction/adduction, "_AA" and thumb/pinky "_CMC_FE")
    joints position the finger lateral spread / thumb opposition angle
    ONCE during FIVE_FINGER_PRESHAPE and are then held fixed while curl
    joints close -- never driven by the closing scalar itself.
"""

from __future__ import annotations

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Functional closing groups (Stage 4 principle: ring+pinky wrap the object
# together to resist rotation, rather than each having independent control
# this early) -- 4 groups per hand, fixed order used throughout this file
# and sharpa_grasp_env.py/the bimanual expert.
GROUPS = ("thumb", "index", "middle", "wrap")
GROUP_FINGERS = {"thumb": ("thumb",), "index": ("index",), "middle": ("middle",), "wrap": ("ring", "pinky")}


def sharpa_joint(side: str, finger: str, suffix: str) -> str:
    return f"{side}_{side}_{finger}_{suffix}"


def sharpa_actuator(side: str, finger: str, suffix: str) -> str:
    return f"{side}_{side}_{finger}_{suffix}_ctrl"


def sharpa_body(side: str, finger: str, link: str) -> str:
    return f"{side}_{side}_{finger}_{link}"


def sharpa_elastomer_geom(side: str, finger: str) -> str:
    return f"{side}_{side}_{finger}_elastomer"


# (finger, suffix) -> role, "curl" or "preshape". Ranges are read from the
# compiled model directly at runtime (sharpa_grasp_env.py), never hard-coded
# here -- this table only says WHICH role each joint plays.
_JOINT_ROLES: dict[str, dict[str, str]] = {
    "thumb": {"CMC_FE": "preshape", "CMC_AA": "preshape", "MCP_FE": "curl", "MCP_AA": "preshape", "IP": "curl"},
    "index": {"MCP_FE": "curl", "MCP_AA": "preshape", "PIP": "curl", "DIP": "curl"},
    "middle": {"MCP_FE": "curl", "MCP_AA": "preshape", "PIP": "curl", "DIP": "curl"},
    "ring": {"MCP_FE": "curl", "MCP_AA": "preshape", "PIP": "curl", "DIP": "curl"},
    "pinky": {"CMC": "preshape", "MCP_FE": "curl", "MCP_AA": "preshape", "PIP": "curl", "DIP": "curl"},
}


def curl_suffixes(finger: str) -> list[str]:
    return [s for s, role in _JOINT_ROLES[finger].items() if role == "curl"]


def preshape_suffixes(finger: str) -> list[str]:
    return [s for s, role in _JOINT_ROLES[finger].items() if role == "preshape"]


# FIVE_FINGER_PRESHAPE targets (radians), held FIXED through closing.
# DISCLOSED, NOT MEASURED: no hardware reference exists for these angles
# (Sharpa has no published grasp-pose dataset this project has access
# to) -- chosen from basic anthropomorphic reasoning (thumb rotates into
# opposition via CMC_FE/CMC_AA; the 4 fingers stay laterally near-neutral,
# pinky's CMC curls it slightly inward to align its row with the others)
# and then VERIFIED empirically this session (scripts/test_sharpa_grasp.py)
# to be self-collision-free and to place the thumb roughly opposing the
# other 4 fingers for a 12cm object -- not assumed correct a priori.
#
# [Level-approach session] Tried CMC_FE 1.05 -> 1.8: an isolated FK sweep
# from a live THUMB_OPPOSE pose showed the thumb tip landing inside the
# object's Z band (+55 to +44mm at +0.7/+0.9), but applying it GLOBALLY
# here changes the thumb's preshape from WRIST_SIDE_GRASP_ALIGN onward
# (set_preshape(1.0) runs there, not just at FIVE_FINGER_PRESHAPE) and
# the full rollout regressed badly -- ALL curl synergies stayed at 0.0
# the entire episode and index's Y offset ballooned to 170-180mm (vs
# ~62-74mm before), meaning the earlier states' own behavior was
# disrupted (most likely a new self-collision from the much-more-folded
# thumb cascading through WRIST_SIDE_GRASP_ALIGN/FOREARM_SIDE_DESCEND).
# Reverted. A real fix needs to apply extra thumb reach locally (e.g.
# only from THUMB_OPPOSE onward) rather than as this shared, always-on
# preshape target -- not done this session.
PRESHAPE_TARGETS: dict[str, dict[str, float]] = {
    "thumb": {"CMC_FE": 1.05, "CMC_AA": 0.0, "MCP_AA": 0.0},
    "index": {"MCP_AA": 0.0},
    "middle": {"MCP_AA": 0.0},
    "ring": {"MCP_AA": 0.0},
    "pinky": {"CMC": 0.15, "MCP_AA": 0.0},
}

# Closing safety margin: curl joints target CLOSE_FRACTION of their real
# ctrlrange span (from the extension/open end), not the hard upper limit,
# leaving headroom before the mechanical stop -- a conservative, disclosed
# choice (not tuned against a specific object), analogous in spirit to
# this project's existing "never command a raw hard limit" convention.
CLOSE_FRACTION = 0.85
