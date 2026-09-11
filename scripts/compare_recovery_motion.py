#!/usr/bin/env python3
"""Compare recovery motion profiles from evaluate_factory_recovery.py outputs.

    python3 scripts/compare_recovery_motion.py results/factory/posture

Reports the numbers that decide whether the approach actually got more direct:
success, time from arrival to first contact and to lift, how much the arms
spread, and whether contact quality moved relative to baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ARRIVAL_PHASE = "SETTLE"
CONTACT_STATE = "CONTACT_ACQUIRE"


def arrival_step(record):
    for event in record["phase_events"]:
        if event["phase"] == ARRIVAL_PHASE:
            return event["step"]
    return None


def first_state_step(record, name):
    for event in record["phase_events"]:
        if event["grasp_state"] == name:
            return event["step"]
    return None


def prepare_repeats(record):
    """How many times the wide clearance pose is entered."""
    return sum(1 for e in record["phase_events"] if e["grasp_state"] == "ARM_LATERAL_CLEARANCE")


def summarise(path: Path) -> dict:
    record = json.loads(path.read_text())
    arrive = arrival_step(record)
    contact = first_state_step(record, CONTACT_STATE)
    return {
        "file": path.name,
        "station": record["station"],
        "profile": record["config"].get("motion_profile", "baseline"),
        "state": record["state"],
        "steps": record["steps"],
        "lift_mm": round(record["max_clearance_m"] * 1000, 2),
        "hold_s": round(record["max_supported_hold_seconds"], 2),
        "penetration_mm": round(record["object_hand_penetration_max_m"] * 1000, 2),
        "forbidden_ticks": record["forbidden_contact_ticks"],
        "fell": record["fallen"],
        "clearance_entries": prepare_repeats(record),
        "arrive_step": arrive,
        "arrive_to_contact": None if (arrive is None or contact is None) else contact - arrive,
        "used_direct_approach": record.get("used_direct_approach"),
    }


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "results/factory/posture")
    rows = [summarise(p) for p in sorted(directory.glob("*.json"))]
    if not rows:
        print(f"no result json under {directory}")
        return 1
    header = ("profile", "st", "state", "steps", "lift_mm", "hold_s", "pen_mm",
              "forbid", "spreads", "arrive", "arr->contact", "direct?")
    print(f"{header[0]:>9} {header[1]:>3} {header[2]:>8} {header[3]:>6} {header[4]:>8} "
          f"{header[5]:>7} {header[6]:>7} {header[7]:>7} {header[8]:>8} {header[9]:>7} "
          f"{header[10]:>12} {header[11]:>8}")
    for r in rows:
        print(f"{r['profile']:>9} {r['station']:>3} {r['state']:>8} {r['steps']:>6} "
              f"{r['lift_mm']:>8.2f} {r['hold_s']:>7.2f} {r['penetration_mm']:>7.2f} "
              f"{r['forbidden_ticks']:>7} {r['clearance_entries']:>8} "
              f"{str(r['arrive_step']):>7} {str(r['arrive_to_contact']):>12} "
              f"{str(r['used_direct_approach']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
