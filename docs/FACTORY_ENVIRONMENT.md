# Two-workcell factory environment — 2026-09-10

## What this is

The environment the future supervisor data will live in: a floating-base
G1 + Sharpa standing between **two separated automation cells**, each with its
own scripted arm and its own part. One cell can fail while the other keeps
producing, and which one fails is decided by the episode seed alone.

This replaces the previous single fixed spot with a single object as the scene
for supervisor work. The fixed-base grasp env is unchanged and still used for
grasp development.

## What this is NOT

**There is no walking controller in this repository.** The audit at the start of
this work confirmed it: the only base-moving code is `PlanarDebugEnv`, which
slides the pelvis on kinematic slide/hinge joints and is labelled in its own
docstring as not a gait. So the G1 does **not** travel between the cells yet.

The base is deliberately left free (`floating_base_joint` is kept, verified as
`mjJNT_FREE`), so when navigation is built it has to come from the legs. Nothing
in this environment moves the base for the robot.

Measured baseline: the G1 stands on its floating base for 2000 steps (20 s) with
0.1 mm of pelvis drift, both feet in contact, no fall. Standing is solid;
walking does not exist.

## Layout

Each workcell is defined in a **local frame whose origin is the pelvis stand
spot for canonical manipulation** and whose +x is the robot's heading. The table
sits at `DEFAULT_TABLE_POS` = (0.30, 0, 0.70) and the part at x = 0.27 in that
frame — exactly the relationship the existing grasp was validated against. Both
cells are therefore the same scene under a rigid transform, and no world
coordinate for a cell is written down twice.

Three spacings were compared before choosing (bearing fixed at ±35° so the
comparison isolates distance; walking speed 0.5 m/s and turn rate 0.6 rad/s for
the tick estimate):

| preset | walk (m) | turn (deg) | gap between table edges (m) | gap between arm columns (m) | nav steps | vs 4194-step grasp episode |
| --- | --- | --- | --- | --- | --- | --- |
| short | 1.50 | 35 | 1.36 | 2.18 | 401 | 9.6% |
| **medium (chosen)** | **2.50** | **35** | **2.51** | **3.29** | **601** | **14.3%** |
| long | 4.00 | 35 | 4.23 | 4.98 | 901 | 21.5% |

`medium` was selected. `short`'s 1.36 m gap leaves the two cells reading as one
cluster and makes "which cell" nearly trivial; `long` costs 50% more ticks for
no additional task content. Tick cost is not the binding constraint at any of
the three — even `long` is a fifth of one grasp episode.

Measured world geometry for `medium`:

- workcell 0 manipulation pose (2.048, −1.434), heading −35°
- workcell 1 manipulation pose (2.048, +1.434), heading +35°
- **separation between the two manipulation poses: 2.868 m**

Everything else per cell is derived from that pose: table, part, arm column,
drop zone, observation pose (0.80 m behind), and the floor marker.

## Automation cells

Each cell has a 3-DoF primitive arm (`base_yaw`, `shoulder_pitch`,
`elbow_pitch`) on a static column, driven by real position actuators. Nothing
teleports a geom per frame. Names are prefixed `wc0_` / `wc1_`, so the two cells
cannot share state.

The scripted cycle is authored in **tip space** (`ARM_CYCLE_TIP_TARGETS`) and
converted to joint angles by closed-form 2-link IK. This matters: the first
version hard-coded joint angles and drove the tip 0.43 m *through* the tabletop.

The clearance that actually binds is against the **part**, not the tabletop, and
it took three measurements to get right:

| pick tip z | arm/part contacts per 800 steps | part drift |
| --- | --- | --- |
| 0.82 | 98 | 62 mm |
| 0.88 | 247 | 136 mm |
| **0.95 (current)** | **0** | **0.000 mm** |

Raising the tip from 0.82 to 0.88 made it *worse*, because the assumed part-top
height (0.81) was wrong — the part rests centred at 0.812 with half size 0.06, so
its top is at **0.872**, and the forearm capsule reaches ~0.030 m below the tip.
`mj_geomDistance` against the compiled model now shows **+48 mm** of minimum
arm/part clearance over the whole cycle. The arm only *mimics* pick and place: it
does not physically transport the part, and it must not disturb it.
`test_arm_cycle_does_not_disturb_the_part` locks both the static clearance and
the running drift.

Compiled contract: `nq=100, nv=97, nu=79` — 73 G1 actuators (the locked
G1+Sharpa contract, unchanged) plus 6 automation-arm actuators.

## Dropped-Part fault

One exception only, as specified.

- The episode seed alone picks the faulting cell and (optionally) the fault
  step. Same seed → same cell, same step, same drop pose, bitwise-identical
  `qpos` after 260 steps.
- At the fault step, that cell's arm stalls at its pick waypoint and its part is
  **released 0.05 m above the drop zone** so it falls and settles under gravity.
  The release itself is scripted fault injection; the landing is real physics
  and is measured.
- The other cell keeps producing. Measured after a fault: the healthy arm sweeps
  0.534 rad while the stalled one sweeps 0.007 rad.

Measured settling (seed 0, 600 steps): rest z **0.8097 m** against an expected
0.8100, `|qvel|max` **0.00000**, contact penetration **0.34 mm**, resting at the
intended drop-zone local xy (0.27, 0.08). Not floating, not sunk, not fallen through.

The drop zone is on the **tabletop**, not the floor. Floor-level picking needs
body lowering, balance under a reaching load, and probably a different grasp
topology than the current two-handed side squeeze (the floor blocks one side).
None of that exists. The zone is a config value so a floor preset can be
evaluated later without another layout change.

## Observation and action

Action is the existing **37-dim whole-body** command (legs 12, waist 3, arms 14,
Sharpa groups 8) — real G1 actuator commands. There is no action that writes the
base pose.

Observation is `WholeBodyEnv`'s own 83 dims plus a **30-dim factory block**
(113 total):

- per cell (×2, 10 each): manipulation pose in base frame (x, y), heading error
  (cos, sin), part position in base frame (x, y, z), arm cycle phase (cos, sin),
  arm fault flag
- then: `fault_active`, target one-hot (2), target part in base frame (x, y, z),
  target manipulation pose in base frame (x, y), target heading error (cos, sin)

Before the fault fires the target one-hot and target block are all zero, so the
answer cannot be read early.

This is the **v1 state-based supervisor**: the target cell is given directly.
A future vision-based supervisor would infer it. No camera is used here.

## Navigation Gate

Thresholds are fixed **now, before any locomotion policy exists**, so they
cannot be relaxed later to manufacture a pass:

| criterion | threshold |
| --- | --- |
| position error at the manipulation pose | ≤ 0.10 m |
| heading error | ≤ 0.15 rad (~8.6°) |
| continuous hold at target | ≥ 100 steps (1.0 s) |
| episode budget | ≤ 1500 steps |
| fall | not allowed |
| pelvis/torso contact | not allowed |
| base displacement per step | ≤ 0.05 m (anything more is not walking) |
| entering the wrong cell first | not allowed |

`NavigationTracker` scores an externally supplied trajectory. It never moves the
robot. Tests confirm it rejects a one-shot base jump, a wrong-cell detour and a
fall, and only scores a gradual kinematic oracle — which is labelled an oracle,
not locomotion, and must never be reported as a walking result.

## Two measured transfer blockers

**1. The grasp Expert cannot run in a cell as written.**
`SharpaBimanualGraspExpert._mirrored_targets` builds approach targets as
`object_pos + [-standoff, ±y_offset, height]` — offsets along **world** axes,
not along the robot's heading. At a 35° cell that is wrong by up to **169 mm**.
It also looks up the object by the fixed name `object`, which does not exist in
the factory model (`wc0_part` / `wc1_part` do). The scene geometry transfers
exactly — both cells reproduce the canonical (0.27, 0) part offset and (0.30, 0)
table offset to 1e-6 — but the Expert needs a heading-frame rewrite first.

**2. The 10 recorded grasp demos are now refused on replay.**
`environment_sha256` covers all of `humanoid_learning/envs/*.py`, so adding the
factory modules changed it (`df5e1133…` → `98c410e8…`) and
`replay_episode` now raises *"environment code differs from recording"*. This is
the guard working as designed. The hash was **not** weakened and no demo file was
edited. A **new dataset version** is required for any factory-era recording.

Follow-up worth considering (not done here): narrow the hash to the modules the
recorded env actually imports, so an unrelated new env file stops invalidating
grasp demos. That is a precision improvement, not a weakening — but it would
itself invalidate the current recordings once, so it should be done together
with the next re-record.

## Gate gap worth knowing about

The Navigation Gate allows **0.10 m** of arrival error. The grasp envelope that
has actually been validated is **±0.010 m** of object XY offset. That is a 10×
gap: arriving inside the Navigation Gate is *not* by itself enough for the
current grasp to succeed. Closing it needs either a wider grasp envelope, a
final alignment step before manipulation, or a tighter navigation threshold.
This is the largest open item.

Also still missing: **Place**. The grasp currently lifts and holds, and releasing
just drops the object. Putting a part back where it belongs is not implemented.

## Running it

```bash
# interactive
DISPLAY=:0 python3 scripts/view_factory.py
DISPLAY=:0 python3 scripts/view_factory.py --seed 1 --fault-workcell 0

# headless stills (green beacon = producing, red = faulted)
OPENBLAS_NUM_THREADS=1 python3 scripts/view_factory.py --offscreen \
  --seed 0 --out results/factory/scene.png --steps 400 --capture 150 400

# tests
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_factory.py
```

Renders go to `results/factory/`, which is git-ignored.
