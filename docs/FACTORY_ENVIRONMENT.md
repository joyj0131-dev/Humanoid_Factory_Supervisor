# Phase 5 — Scripted factory automation environment

Conveyor line with two automation stations. Last updated 2026-09-10.

**Phase status: active, NOT complete.** The environment, the fault scenarios and
the recovery signalling are built and tested (21/21). Contact quality is not
resolved (see the full-cycle audit below), and no recovery skill, walking
controller or IL/PPO work has started -- those are Phase 6 and later.

> Sequencing note, recorded so it is not mistaken for an oversight:
> `docs/GIT_WORKFLOW.md` rule 5 says Phase 5 does not begin until Phase 4.5's
> Gate A→D all pass, and Gate A (thumb opposition) still does not. The user
> directed this environment work explicitly. What that rule guards -- whole-body
> locomotion and IL/BC/PPO -- has **not** been started here; only the scripted
> factory scene the later phases will need. Phase 4.5's grasp gates are
> unchanged and still open.

## What this is

The environment the future supervisor data will live in: a floating-base
G1 + Sharpa beside **one conveyor with two stations**, each with its
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

The values in this historical tuning section cover the original experiments.
The full-cycle audit below supersedes them for current contact-quality claims.

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

### Current full-cycle contact audit and smoother jaws

Jaw targets now follow smoothstep over 60% of each dwell, like the arm joints,
instead of changing instantaneously. No mass, friction, contact stiffness,
actuator strength or collision mask changed. Measured over 960 steps (two
cycles), seed 0, both stations reached the outfeed:

| metric | previous step jaws | smooth jaws |
| --- | --- | --- |
| station 0 peak jaw/part force | 54.42 N | 45.26 N |
| station 1 peak jaw/part force | 66.14 N | 53.83 N |
| station 0 peak penetration | 47.80 mm | 43.31 mm |
| station 1 peak penetration | 48.85 mm | 46.25 mm |

These are samples after every environment step across ALL phases, not substep
maxima or hold-only values. The old approximately 10 mm figure is not a bound
over production cycles. Contact quality remains poor despite lower force;
outfeed reach does not establish reliable production or clean grasping.
Higher-priority stiffer contact candidates reduced penetration to 13–17 mm but
raised peak force to 85–97 N and were rejected. Belt jerk is unchanged.

Reproduce the comparison (each prints JSON):

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/audit_factory_grip.py --step-jaws
OPENBLAS_NUM_THREADS=1 python3 scripts/audit_factory_grip.py
```

## Line loop and the Dropped-Part fault

### Two factory-only scenarios (current)

The user explicitly selected two environment scenarios before any humanoid
controller work. Recovery skill training remains separate and has not started.

- `dropped_part`: wait for both jaws to contact the selected part, measured lift
  above 25 mm and transfer over 100 mm; hold the arm and open its jaws. Raise a
  fault only after jaw contacts disappear and the part falls over 20 mm.
  No runtime object qpos/velocity rewrite is used. It lands on the conveyor.
- `misplaced_part`: at reset place the selected part 240 mm downstream of the
  pick pose with yaw 0.30 rad. The belt transports it normally. After the arm's
  nominal grip attempt, a supported part over 100 mm from the pick position and
  zero jaw contacts triggers `pick_failed`. This is initial scene placement,
  not runtime teleportation. It tests combined position/yaw error, not every
  possible yaw-only failure.

`fault_step` is the earliest eligible detection/injection step, not a promise
that the fault occurs at that exact time. If a physical prerequisite is not met,
the scenario stays waiting rather than fabricating an event. Reset aligns the
selected station with the beginning of its cycle, with the other station offset.

`info['recovery_task']` contains station, fault_type, part_body, actual position,
target position and target yaw. Both request returning the part to its nominal
pick pose. Recovery checks position, orientation (cube quarter-turn symmetry),
belt support and settled velocity before restart. The faulted arm restarts a
fresh approach cycle instead of resuming the interrupted grip command.

Both scenarios × both stations passed physical detection and the task-manager
recovery loop. Test code restores the part and sends the handshake; **G1 does
not perform that restoration**. Tests prohibit `_set_part` during scenario
execution and reject a returned-but-still-rotated part. A large-angle/no-contact
pick miss is detected; arbitrary real-world fault detection is not claimed.

```bash
DISPLAY=:0 python3 scripts/view_factory.py --scenario dropped_part
DISPLAY=:0 python3 scripts/view_factory.py --scenario misplaced_part --fault-workcell 0
DISPLAY=:0 python3 scripts/view_factory.py --no-fault
```

The following loop description applies to both faults. Older references to
placing a part above a preselected drop zone are superseded by physical release.

A **physical stop blade** just downstream of each pick spot indexes parts, so
the belt can keep running while a part waits. Normal production is a real closed
loop: the arm picks the part off the belt, sets it down 0.32 m upstream, and the
running belt carries it back against the blade.

The task manager (`FactoryTaskManager`) is the signalling layer:

```
RUNNING -> FAULT_RAISED -> RECOVERING -> RECOVERY_VERIFIED -> RUNNING
           request         accept +      measured pose       restart
           line stopped    complete      stable 0.5 s
```

- The episode seed alone picks which station fails and when.
- On measured fault **the belt stops** (line stop), and the target
  station is published to the observation.
- The supervisor must accept the correct station's request, then submit a
  completion request. Position alone and a completion request alone cannot restart.
- Recovery is judged from the part's **measured pose** -- back at the pick spot,
  settled, held for 0.5 s after completion was requested. It is checked again at
  the restart boundary; the belt stays stopped throughout verification.
- Only then does the manager signal back: belt restarts and the stalled arm
  resumes.

The earlier automatic pose-only restart has been replaced by the explicit
handshake. `env.step(action, supervisor_signal=("accept", station))` accepts a
request; `("complete", station)` requests verification. Incorrect stations and
out-of-order messages return `info["supervisor_signal_accepted"]=False`.
`info["mission_events"]` records request, acceptance, verification and restart.
These messages belong in future factory demos alongside the motor commands;
the 37 motor commands alone do not describe the full mission.

**Recovery cannot happen on its own.** No policy here can walk to a station and
move a part. In tests the restoration is done by the test harness and is labelled
as such. The loop exists so it closes once a policy can do it.

## Observation and action

Action is the existing **37-dim whole-body** command (legs 12, waist 3, arms 14,
Sharpa groups 8) — real G1 actuator commands. There is no action that writes the
base pose.

Observation is `WholeBodyEnv`'s own 83 dims plus a **39-dim factory block**
(122 total, revised factory schema):

- per cell (×2, 10 each): manipulation pose in base frame (x, y), heading error
  (cos, sin), part position in base frame (x, y, z), arm cycle phase (cos, sin),
  arm fault flag
- then: `fault_active`, target one-hot (2), target part in base frame (x, y, z),
  target manipulation pose in base frame (x, y), target heading error (cos, sin),
  `belt_running`
- mission state one-hot (4), verification progress, completion-request flag (6)
- detected fault type one-hot (2), all zero before detection

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

The environment starts navigation timing only when a fault is raised. Waiting
for production to fail does not consume the travel budget. The displacement
check detects large jumps; it cannot prove that a smooth trajectory is physical
walking (a gradual kinematic oracle also satisfies it).

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

The viewer now shows a live status panel, station phases, target and verification
progress. Green beacons mean production; red means a pending request; amber
means accepted recovery/verification. SPACE pauses, R resets, 0 shows the whole
line, and 1/2 focus a station. A accepts the current request and C requests
verification: these are explicitly manual task-manager inputs, not G1 recovery.
Neither key moves any part. At episode end the scene remains visible until R.
Offscreen PNGs include the same status panel.

## Phase 5 step 1 — grasping while standing on two feet

The factory G1 stands on its own legs, but every grasp result in this project
was measured with the pelvis **welded to the world**. Replaying the identical
Expert commands with the base free makes the robot fall over: pelvis drops
753 mm, rolls 179 deg, down at step 839 during the reach.

`GraspEnvConfig.fixed_base` (default `True`, so nothing existing changes) selects
the free-base build.

### The cause is dynamic, not a balance margin

A first measurement suggested the CoM only had 12 mm of forward margin and
crossed the toe line during the reach. **That measurement was wrong**: it used
`subtree_com[0]`, which includes the table and the object. Using the robot's own
CoM (`subtree_com[pelvis]`):

| | robot CoM x | toe line | margin |
| --- | --- | --- | --- |
| standing | +0.0030 | +0.1250 | **122 mm** |
| peak during a successful grasp | +0.0484 | +0.1250 | **77 mm** |

So the CoM never leaves the support polygon, and a wider or braced stance --
which the wrong number would have led to -- fixes nothing. What actually happens
is that the pelvis **rocks with growing amplitude** (pitch −0.4 → −3.2 → +5.6 →
−3.9 deg) until the left foot's normal force reaches 0.00 N and it topples. The
Expert's arm trajectories were tuned against a pelvis that silently absorbed
their reaction torque.

Leg gains are identical in the grasp and whole-body envs (kp 500), so the legs
were never the difference; the arms are (kp 120 compliant vs 500).

### Fix: a minimal ankle-strategy stabiliser

`humanoid_learning/expert/stance_stabilizer.py` reads pelvis roll/pitch and
their rates and commands the four ankle actuators to oppose them. It writes only
the ankles -- arms, hands, waist and the env action are untouched.

Gain sweep against the full grasp (seed 0):

| pitch kp | result | peak pitch | pelvis drop |
| --- | --- | --- | --- |
| none | FAILURE, fell | 89.7 deg | 753 mm |
| **1.0 (chosen)** | **SUCCESS** | **3.0 deg** | **0.3 mm** |
| 2.0 | grasp gate failed (stood) | 3.0 deg | 0.5 mm |
| 3.0 | grasp gate failed (stood) | 2.9 deg | 1.7 mm |
| 4.0 | FAILURE, fell | 89.5 deg | 756 mm |
| 5.0 | FAILURE, fell | 89.7 deg | 752 mm |

Too much gain oscillates the robot over, which is why the value is low.

### Result

Seeds 0/1/2, free base, stabiliser at kp 1.0: **SUCCESS in all three**, object
clearance 80.78 mm against the fixed-base 81.00 mm, peak pitch 3.02 deg, pelvis
drop 0.33 mm. The grasp quality is essentially unchanged by standing free.

`scripts/test_free_base_grasp.py` (4/4) locks all of it, including that the
unstabilised case really does topple, so the stabiliser cannot be quietly
removed.

### Still missing for step 1

**Place is not implemented.** The robot grasps and lifts while standing, but it
still cannot put the block down at a commanded pose and let go -- releasing just
drops it. That is the remaining half of step 1.

The stabiliser cannot take a step, so it cannot recover from a disturbance that
needs one. It is not a walking controller and must never be reported as one.

## Walking to the stations — SOLVED with an external policy (2026-09-10)

**Result: the G1 walks from home to either station and passes the Navigation
Gate.** Final position error 34.8 mm (station 0) and 46.8 mm (station 1) against
the 100 mm limit, heading within 4.5 deg of the 8.6 deg limit, held for 641/579
steps against the 100 required, no falls, no forbidden contact. Maximum base
displacement per step is 4.7 mm -- real gait, far under the 50 mm teleport
threshold the gate rejects.

Locomotion is **Unitree's pre-trained G1 policy** (`unitree_rl_gym`, BSD
3-Clause), used unmodified. It is an external tool, not part of this project's
research: the BC vs BC+PPO comparison applies to the recovery skill, and walking
is treated as given. Provenance, license and checksum are in
`assets/policies/g1_walk/NOTICE`; the binary is not committed and
`scripts/install_g1_walk_policy.py` reproduces it.

```bash
python3 scripts/install_g1_walk_policy.py
DISPLAY=:0 python3 scripts/view_factory.py --walk-to 0
DISPLAY=:0 python3 scripts/view_factory.py --walk-to 1 --scenario dropped_part
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_factory_walk.py
```

### Integration, and four things that had to be fixed

The policy contract was read from `deploy_mujoco.py` and `g1.yaml`: 47 obs (base
angular velocity, gravity in base frame, velocity command, leg pos/vel, previous
action, 0.8 s gait phase), 12 leg actions scaled by 0.25 onto a crouched default
stance, at 50 Hz with kp=[100,100,100,150,40,40], kd=[2,2,2,4,2,2].

1. **Control mode.** This env drives legs with position actuators at kp=500. A
   MuJoCo position actuator with `gainprm=[kp]`, `biasprm=[0,-kp,-kv]` computes
   exactly `kp*(ctrl-q) - kv*qvel` -- the policy's own PD law -- so the gains are
   retuned rather than a torque path bolted on.
2. **Indexing.** Unitree's scene is legs-only and uses `qpos[7:]`; this robot has
   a waist, arms and two hands, so leg indices are resolved by joint name.
3. **Steady-state error.** The policy under-tracks small commands on this robot,
   leaving a ~70 mm shortfall. Integral action (clamped, and only within 0.6 m)
   removes it.
4. **It cannot stand still.** Commanding zero velocity still produces gait steps
   and drifts 0.6 m over 15 s. Arriving is therefore a **handoff**: the leg
   actuators go back to stiff position gains and hold. Freezing mid-gait left
   the robot leaning and 90 mm off, so the handoff waits for **double support**
   (both feet above 40 N). With that, drift after arrival is under 57 mm.

### A real layout bug this exposed

The first walking runs failed the gate on `forbidden_contact`, and it was
correct: the **torso struck the conveyor's near side rail**. The manipulation
pose is at x=1.70 and the rail spanned x 1.710-1.750 at z 0.750-0.810, while the
torso bottom sits at z=0.817 and its front reaches x=1.779 -- 7 mm of clearance,
which walking bob closed. The rails were added for looks and were blocking the
robot from the spot it must stand on. Rail height is now 0.030 m (top at 0.780),
which still catches a part whose underside is at 0.752 while clearing the torso
by 37 mm.

### Sim-to-sim gap, stated not hidden

The policy was trained on a stock G1. This robot carries **2.49 kg of Sharpa
hands (7% of its mass)** that the policy never saw, plus a waist and arms its
training scene did not have. It walks anyway, but under-tracks velocity
commands, which is why the navigator needs integral action.

### What was tried first and abandoned

#### From-scratch attempt (kept for the record)

Both stations sit at heading 0, the same as the G1's home pose, so reaching
either one needs forward + sideways stepping and **no turning** -- the easy case.
Distance is 2.02 m, and the Navigation Gate allows 1500 steps (15 s), so ~0.135
m/s is required.

It does not work, and the blocker is measured rather than guessed.

**Stepping needs full single support.** Progress toward it:

| approach | result |
| --- | --- |
| open-loop lateral lean | falls; the "single support" seen at 120 steps was a transient mid-topple, gone by 400 |
| closed-loop CoM tracking | **does not fall**; weight transfers smoothly to ~84-90% on one foot |
| pushing to 100% transfer | **falls every time**, at every gain and target tried |

So the robot can put 90% of its weight on one foot and hold it, but the instant
the other foot's normal force reaches 0.00 N it topples.

**Why, from the model:** the robot is 35.8 kg (352 N) and the foot sole is only
**37.8 mm half-width**. Ankle torque is NOT the limit -- a kp=500 position
actuator delivers the required 13.3 N-m at 0.027 rad of tracking error. The limit
is the support margin: in single support the CoM must stay inside +-37.8 mm while
ankle roll is capped at +-0.262 rad. A single-axis CoM-y PD is not tight enough
to hold that.

**What walking actually needs** (none of it present): simultaneous CoM x and y
regulation, a swing-leg trajectory that does not disturb the CoM, and ZMP or
capture-point feedback -- or a learned policy. That is a controls project in its
own right, not controller tuning.

Watch it fail:

```bash
DISPLAY=:0 python3 scripts/view_whole_body.py --weight-shift --no-restart
```

Nothing here fakes the result. The Navigation Gate still rejects base
teleportation, `PlanarDebugEnv` is still labelled a kinematic oracle, and no
navigation result is claimed.

## Correction (2026-09-10): the earlier error analysis below was wrong

Two claims in the section that follows were mistaken and are corrected here.
The section is kept because the workspace-window sweep itself stands.

**1. "35-47 mm outside the window laterally" was wrong.** That compared a TOTAL
position error against a LATERAL tolerance. Decomposing the arrival error in the
robot's own frame:

| | station 0 | station 1 |
| --- | --- | --- |
| fore/aft error | +34.3 mm | +44.0 mm |
| **lateral error** | **−11.1 mm** | **+7.4 mm** |

Lateral is at the edge of the +-10 mm window, not 3-5x outside it. The binding
axis is **fore/aft**: the robot stops short, so the block sits 304-314 mm ahead
instead of 270 mm, and +35 mm fore/aft is exactly where the sweep shows
CONTACT_LOST. So the conclusion "widen the grasp laterally" did not follow.

**2. The stance stabiliser was a silent no-op in the factory.** It wrote the
ankle `data.ctrl`, and `WholeBodyEnv.step()` overwrites every leg actuator from
its internal `_leg_target` afterwards. Measured: the ankle command went 0.3 to
0.0 across a single step. It worked in `SharpaGraspEnv` only because that env
never writes leg ctrl. Any claim that the free-base grasp result carried over to
the factory was unfounded.

Also corrected: the sweep measures the **current controller's** success envelope,
not a kinematic limit of the arm. `OBJECT_MOVED_TOO_MUCH`, `CONTACT_LOST` and
`TIMEOUT` are approach and contact failures. Calling it "the arm's physical
limit" overstated it.

### What was fixed

- The stabiliser now also writes `_leg_target`, so its command survives the
  env's rewrite. It is **scoped to the grasp phase**: after the walk handoff the
  legs are stiff-held, which already stands fine, and running the regulator
  there made the robot fall (its gains are tuned for the compliant grasp plant).
- `g1_walk_policy.stand_pose_for_part()` derives the stand pose from the
  **measured part position** instead of the station's nominal spot. With the
  part in the drop zone the nominal aim leaves a −207.5 mm lateral error;
  aiming at the part brings it to **+8.6 mm**, inside the window.

### What is still open

Fore/aft. The navigator latches arrival on total distance and stops 34-44 mm
short. A per-axis test (fore 20 mm / lateral 12 mm) was tried and reverted: this
controller's closest approach is ~50 mm total, so it never latched and the robot
wandered off. Closing the fore/aft gap needs a more precise final approach, not
a tighter threshold on the existing one.

## Grasping at the station — BLOCKED by the grasp's workspace window

Walking works, but "walk there, then grasp" does not, and the reason is measured.

The walking policy arrives **35-47 mm** from the stand spot. The grasp needs the
part far closer to its canonical position than that. Sweeping the part offset
(`scripts/audit_grasp_workspace_window.py`):

| offset from the canonical spot | result |
| --- | --- |
| 0 | SUCCESS |
| −30 mm fore/aft (nearer) | SUCCESS |
| **+20 mm lateral** | **FAILURE** (OBJECT_MOVED_TOO_MUCH) |
| ±35 mm lateral | FAILURE (TIMEOUT) |
| +35 mm fore/aft (further) | FAILURE (CONTACT_LOST) |
| −50 mm or more (too near) | FAILURE (FORWARD_REACH_NOT_ACHIEVED) |

Usable window: roughly **dx −30…+10 mm, dy −10…+10 mm**. Arrival error is 3-5x
outside it laterally.

**This is the grasp's limit, not the free base's.** A welded pelvis fails at the
same offsets, so the stabiliser and the walking are not implicated.

**Why it is so narrow.** The approach targets already follow the object
(`_mirrored_targets` is object-relative), so this is not a framing bug. The
Expert's own notes record that a single solve to the approach pose "drove
waist_pitch to its hard limit ... a real reachability property". The bimanual
grasp is already operating at the edge of the arm's reachable workspace, so it
has no lateral slack to spend on positioning error.

**Standing closer does not buy slack** — tested, because it was the obvious
guess: at −30 mm a lateral 30 mm still fails, and at −50 mm the approach cannot
be executed at all.

### What this rules out

Loosening the Navigation Gate is the wrong direction: more arrival slack makes
this worse, not better. Something has to close the 35-47 mm gap:

1. **Widen the grasp's lateral tolerance.** The right kind of work for this
   project, and squarely in the area it studies. Largest effort.
2. **Fine positioning after arrival** — translate the pelvis with both feet
   planted. A first attempt at this collapsed the robot into a squat (pelvis
   0.79 → 0.59 m): translating the pelvis while keeping it upright and the feet
   flat needs coordinated hip/knee/ankle, not a two-joint lean.
3. **Corrective side-steps** — the walking policy can side-step, but with the
   same ~50 mm precision, so this may not converge.

No option is implemented. Walking and grasping both work; they do not yet
compose.

## (a) Fore/aft precision — characterised, NOT solved

Attempted the precise final approach that would put the block inside the grasp
window. It is not solved. What was measured:

**The window is never entered.** Tracing the approach with latching disabled,
the fore/aft error bottoms out at +83.9 mm (station 0) and +38.6 mm (station 1)
against a +10 mm upper bound. Zero ticks out of 1600 land inside the window, so
opportunistic latching cannot work either.

**A zero velocity command does not hold position.** Measured command versus
achieved forward velocity on this robot:

| commanded vx | achieved m/s |
| --- | --- |
| 0.00 | **−0.0392** (walks backward) |
| 0.02 | −0.0217 |
| 0.05 | +0.0034 |
| 0.08 | +0.0275 |
| 0.12 | +0.0602 |
| 0.25 | +0.1648 |
| 0.35 | +0.1722 (saturating) |

The zero crossing is at about **0.047**: that much command is simply the price of
standing still. As the goal is approached the command shrinks toward zero, the
backward drift takes over, and the robot stalls short. `forward_command_bias`
now compensates it, though near the goal the command is pinned at the speed
floor so the bias is currently inert there.

**The stall is systematic, not gait noise.** It sits at +49.7/+47.6 mm with only
about ±5 mm of ripple, and raising the near-goal speed floor moves it to
+37/+36 mm before getting worse again at a floor of 1.0.

| near-goal speed floor | station 0 | station 1 |
| --- | --- | --- |
| 0.25 (default) | +49.7 mm | +47.6 mm |
| 0.40 | +42.9 mm | +40.9 mm |
| 0.60 | +37.4 mm | +36.2 mm |
| 1.00 | +36.6 mm | +45.4 mm |

**What was tried and reverted.** Shifting the goal forward by the measured stall
looked obvious given how repeatable it is, but it fires the distance-based
arrival latch at a different point on the approach: station 1's lateral error
went from +7 mm to +36 mm. Compensating the stall needs the arrival criterion
reworked first, not a shifted target bolted onto the existing one. Likewise a
per-axis arrival test was reverted earlier because this controller cannot meet
it and then never latches at all.

**Where this leaves it.** Roughly 36-50 mm of fore/aft has to come from
somewhere other than the current walk-and-latch loop. The three candidates are
a redesigned final-approach phase with its own arrival criterion, a non-stepping
fine adjustment, or letting the grasp absorb a further +40 mm (it currently
fails at +35 mm). Walking itself is unaffected and still passes 5/5.
