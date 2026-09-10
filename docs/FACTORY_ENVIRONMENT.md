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

## Layout: one conveyor, two stations

One straight conveyor runs along world +Y at x = 2.0 with its surface at
z = 0.75 -- the same working height as the canonical grasp table. Two arm
stations sit on it, **both facing world +X**, so the G1 approaches each of them
exactly as it approaches the canonical fixed-base grasp scene.

That straight-line choice removed the previous layout's biggest blocker. Under
the earlier +-35 degree workcell placement, the grasp Expert's world-axis
approach offsets were wrong by up to 169 mm at a station. With both stations
facing the same way they differ by a pure translation, and those offsets are now
exact (`test_straight_line_layout_keeps_the_grasp_experts_world_axis_offsets_valid`).

| preset | station spacing | home -> station | both face |
| --- | --- | --- | --- |
| short | 1.6 m | 1.88 m | +X |
| **medium (default)** | **2.2 m** | **2.02 m** | **+X** |
| long | 3.0 m | 2.27 m | +X |

Station-local geometry is the canonical grasp relationship verbatim: belt centre
0.30 m ahead of the pelvis, part at 0.27 m. Both stations reproduce it to 1e-6.

## Automation stations

Each station has a 4-DoF arm (`base_yaw`, `shoulder_pitch`, `elbow_pitch`,
`wrist_pitch`) on a static column across the belt from the robot, plus a **real
two-jaw gripper** on slide joints with friction. `wrist_pitch` is solved so the
three pitches always sum to pi/2, holding the gripper vertical so the jaws
descend onto a part from above. The cycle is authored in tip space and converted
by closed-form 2-link IK; the grasp pose lands within 0.005 m of the part centre.

Compiled contract: `nq=106, nv=103, nu=85` -- 73 G1 actuators (unchanged) plus
12 station actuators.

The eight-step cycle is: approach, descend, grip, lift, traverse, lower,
release, retract. Targets **ramp** between waypoints (smoothstep over the first
60% of each dwell, then settle). Stepping the targets instead flung the gripped
part off the line during the rotation to the outfeed.

### Three real bugs found here, all by measurement

**1. The arm jammed against its own column.** MuJoCo's `filterparent` does not
exclude an arm link from its column, because the column has no joint and is
therefore welded to the world -- the pair reads as link-vs-world, which
`filterparent` deliberately never filters. The turret collided with its own
column at 2.6e17 N and pinned `base_yaw` at 0.50 rad while it was commanded to
0. Fixed by putting every arm geom in `contype=2 / conaffinity=1`, so the arm's
links are transparent to each other but still collide with the part, belt, floor
and robot.

**2. Gripper force is not what it looks like.** A position actuator's grip force
is `kp x (target - actual)`, so a jaw that stalls 1 mm from its target pushes
with barely 1 N no matter how deep the nominal squeeze looks. Measured sweep at
kp = 1200:

| closed target | peak jaw force | contact penetration | result |
| --- | --- | --- | --- |
| 0.063 | 3.1 N | 3.6 mm | dropped |
| **0.058 (current)** | **11.6 N** | **10.1 mm** | **lifted** |
| 0.054 | 22.8 N | 12.2 mm | lifted |
| 0.050 | 24.3 N | 15.3 mm | lifted |

An earlier 18 mm squeeze at kp = 250 drove 9.3 mm of penetration, let the part
slip 70 mm through the jaws during the lift, and then **ejected it upward** --
the same soft-contact "squirt" failure this project already recorded for the
Sharpa hand. Stiffening contact to `solref=(0.002, 1)` cut penetration but needed
~59 N to still lift, which is worse. 10 mm of penetration on a 120 mm part is
real and is not claimed to be a clean grasp; this is a scripted factory prop,
held to a lower bar than the G1's own Sharpa grasp.

**3. The belt could not move anything.** The friction holding a part on the belt
is mu*m*g = 1.5 * 0.1 * 9.81 = 1.47 N; the first drive gain produced 0.30 N, so
parts simply never moved. The gain now produces up to 5 N (clipped), and the
drive is applied with the matching `r x F` torque so it acts at the contact
patch rather than the centre of mass -- applying it at the COM alone tipped the
part and made it hop along the belt.

Transport is functional but jerky: a part travels the 0.32 m from outfeed back to
the stop blade in about 3 s, with velocity fluctuating over roughly 0 to 0.3 m/s.
This is a viscous surface-drive approximation, not a simulated belt mechanism.

## Line loop and the Dropped-Part fault

A **physical stop blade** just downstream of each pick spot indexes parts, so
the belt can keep running while a part waits. Normal production is a real closed
loop: the arm picks the part off the belt, sets it down 0.32 m upstream, and the
running belt carries it back against the blade.

The task manager (`FactoryTaskManager`) is the signalling layer:

```
RUNNING  --fault-->  FAULT_RAISED            RECOVERY_VERIFIED --> RUNNING
                     belt STOPPED             belt + arm restart
                     arm stalled
                     G1 called (target set)
```

- The episode seed alone picks which station fails and when.
- On fault the arm's jaws open, the part ends up past the stop blade where the
  arm's own cycle never reaches, **the belt stops** (line stop), and the target
  station is published to the observation.
- Recovery is judged from the part's **measured pose** -- back at the pick spot,
  settled, held for 0.5 s -- never from a flag the robot sets.
- Only then does the manager signal back: belt restarts and the stalled arm
  resumes.

Measured (seed 0): fault at step 200 stopped the line; the line did **not**
restart on its own through 150 further steps; after the part was restored the
manager verified and restarted belt and arm at step 464.

**Recovery cannot happen on its own.** No policy here can walk to a station and
move a part. In tests the restoration is done by the test harness and is labelled
as such. The loop exists so it closes once a policy can do it.

## Observation and action

Action is the existing **37-dim whole-body** command (legs 12, waist 3, arms 14,
Sharpa groups 8) — real G1 actuator commands. There is no action that writes the
base pose.

Observation is `WholeBodyEnv`'s own 83 dims plus a **31-dim factory block**
(114 total):

- per cell (×2, 10 each): manipulation pose in base frame (x, y), heading error
  (cos, sin), part position in base frame (x, y, z), arm cycle phase (cos, sin),
  arm fault flag
- then: `fault_active`, target one-hot (2), target part in base frame (x, y, z),
  target manipulation pose in base frame (x, y), target heading error (cos, sin),
  `belt_running`

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

## Transfer status

**The grasp Expert's world-axis problem is resolved** by the straight line: both
stations face world +X, so `object_pos + [-standoff, +-y, h]` is correct at each.
It still looks the object up by the fixed name `object`, which does not exist in
the factory model (`wc0_part` / `wc1_part` do), so it needs that one change
before it can run here.

**The recorded grasp demos remain refused on replay.** `environment_sha256`
covers all of `humanoid_learning/envs/*.py`, so the factory modules changed it
and `replay_episode` raises *"environment code differs from recording"*. The hash
was not weakened and no demo file was edited. A new dataset version is required.

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
  --seed 0 --out results/factory/line.png --steps 400 --capture 120 260

# tests
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_factory.py
```

Renders go to `results/factory/`, which is git-ignored.
