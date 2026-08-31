"""Block Size/Position Feasibility Map sweep driver (Phase 4 Grasp Track,
Part A). Runs humanoid_learning.expert.grasp_feasibility_map.run_condition
across a grid of object half_size/x/y with the UNCHANGED canonical
controller and writes a CSV + JSON report. Run with:

    python scripts/run_grasp_feasibility_map.py --coarse
    python scripts/run_grasp_feasibility_map.py --fine --center-half-size 0.06 --center-x 0.27 --center-y 0.0
    python scripts/run_grasp_feasibility_map.py --repeat --half-size 0.06 --x 0.27 --y 0.0 --reps 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from humanoid_learning.expert.grasp_feasibility_map import condition_to_row, run_condition  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "grasp_feasibility_map"

COARSE_FULL_SIZES_CM = [11.5, 12.0, 12.5, 13.0]
COARSE_X = [0.25, 0.26, 0.27, 0.28, 0.29]
COARSE_Y = [-0.02, -0.01, 0.00, 0.01, 0.02]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in rows:
        fr = dict(r)
        fr["left_group_contact"] = "".join("1" if v else "0" for v in fr["left_group_contact"])
        fr["right_group_contact"] = "".join("1" if v else "0" for v in fr["right_group_contact"])
        fr["thumb_first_contact_obj_xy"] = json.dumps(fr["thumb_first_contact_obj_xy"])
        fr["state_entry_step"] = json.dumps(fr["state_entry_step"])
        flat_rows.append(fr)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def run_grid(sizes_cm: list[float], xs: list[float], ys: list[float], seed: int, tag: str) -> list[dict]:
    rows = []
    total = len(sizes_cm) * len(xs) * len(ys)
    n = 0
    t0 = time.time()
    for size_cm in sizes_cm:
        half = round(size_cm / 200.0, 6)
        for x in xs:
            for y in ys:
                n += 1
                r = run_condition(half, x, y, seed=seed)
                row = condition_to_row(r)
                rows.append(row)
                print(
                    f"[{n}/{total}] size={size_cm}cm x={x} y={y} -> "
                    f"state={row['final_state']} tripod={row['max_bilateral_tripod_streak']} "
                    f"multifinger={row['max_bilateral_multifinger_streak']} "
                    f"xy_disp={row['object_xy_displacement']} ang_rms={row['angular_speed_rms']} "
                    f"gate_a={row['gate_a_pass']}"
                )
    print(f"{tag}: {n} conditions in {time.time() - t0:.1f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds for --repeat")
    ap.add_argument("--center-half-size", type=float, default=0.06)
    ap.add_argument("--center-x", type=float, default=0.27)
    ap.add_argument("--center-y", type=float, default=0.0)
    ap.add_argument("--half-size", type=float, default=0.06)
    ap.add_argument("--x", type=float, default=0.27)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    if args.coarse:
        rows = run_grid(COARSE_FULL_SIZES_CM, COARSE_X, COARSE_Y, args.seed, "coarse")
        write_csv(rows, RESULTS_DIR / "coarse_map.csv")
        print(f"wrote {RESULTS_DIR / 'coarse_map.csv'}")
    elif args.fine:
        c_half = args.center_half_size
        fine_sizes_cm = [round((c_half + d) * 200.0, 4) for d in (-0.0025, -0.00125, 0.0, 0.00125, 0.0025)]
        fine_x = [round(args.center_x + d, 5) for d in (-0.005, -0.0025, 0.0, 0.0025, 0.005)]
        fine_y = [round(args.center_y + d, 5) for d in (-0.005, -0.0025, 0.0, 0.0025, 0.005)]
        rows = run_grid(fine_sizes_cm, fine_x, fine_y, args.seed, "fine")
        write_csv(rows, RESULTS_DIR / "fine_map.csv")
        print(f"wrote {RESULTS_DIR / 'fine_map.csv'}")
    elif args.repeat:
        seeds = [int(s) for s in args.seeds.split(",")]
        rows = []
        for seed in seeds:
            for _ in range(args.reps if seed == seeds[0] else 1):
                r = run_condition(args.half_size, args.x, args.y, seed=seed)
                row = condition_to_row(r)
                rows.append(row)
                print(f"seed={seed} -> state={row['final_state']} tripod={row['max_bilateral_tripod_streak']} "
                      f"xy_disp={row['object_xy_displacement']} gate_a={row['gate_a_pass']}")
        write_csv(rows, RESULTS_DIR / "repeat_check.csv")
        print(f"wrote {RESULTS_DIR / 'repeat_check.csv'}")
    else:
        ap.error("pass --coarse, --fine, or --repeat")


if __name__ == "__main__":
    main()
