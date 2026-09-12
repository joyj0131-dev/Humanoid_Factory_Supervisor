# Factory fault-to-lift integration (2026-09-11)

## Continuous approach (2026-09-12)

The viewer's `--recover` now selects `--recovery-motion smooth`.
`--recovery-motion direct` preserves the previous 274811c motion for comparison;
`baseline` and `compact` remain available. The standalone grasp default is unchanged.

The old ALIGN emitted a new IK target every 30 control ticks (0.30 s).
DESCEND solved IK every tick but still held its Cartesian position target for
30 ticks. Smooth mode uses a single quintic phase clock over each whole segment:
position and ALIGN orientation advance every tick without intermediate waypoint
stops. DESCEND's existing wrist trim is blended once over its initial 0.30 s,
not jumped or restarted at subsequent waypoint boundaries. IK still uses measured
joint positions and actions go through the same real physics/control path.

This is **not** elimination of every pause: the walking handoff, endpoint posture
checks, preshape check, contact settling and five-second supported lift remain.
The raised-arm preparation before walking also remains. Segment durations and
success/contact thresholds were not shortened. Smoothness is not a speedup claim.

The evaluator records actual palm speed metrics by phase (interior 80% of the
phase, slow = below 2 mm/s per hand) and controller wall-time p50/p95. This separates
physical stop/start motion from rendering/computation delays; real-time GUI frame
rate is not guaranteed by a successful headless rollout.

Final measured results (seed 0, dropped_part, same scene; **not generalization**):

| Metric | direct st0 | smooth st0 | direct st1 | smooth st1 |
| --- | --- | --- | --- | --- |
| Lift + supported hold | PASS | PASS | PASS | PASS |
| Steps | 5179 | 5171 | 5292 | 5592 |
| Max vertical clearance, mm | 79.92 | 80.20 | 73.08 | 55.75 |
| Continuous support, s | 5 | 5 | 5 | 5 |
| Max hand/object penetration, mm | 5.31 | 5.12 | 6.19 | 4.82 |
| Forbidden-contact ticks | 938 | 989 | 840 | 847 |

Evidence: `results/factory/continuous_final_st{0,1}.json`, before st0
`continuous_before_st0.json`; direct st1 is the previous documented 274811c
result. For st0 ALIGN, interior palm speed standard deviation left/right falls
from 0.02194/0.02408 to 0.01355/0.01475 m/s (about 38%). DESCEND falls from
0.03211/0.03011 to 0.02340/0.01592 m/s. These are variation measurements, not
a proof of jerk-free motion: ALIGN peak speeds increase from 0.112/0.124 to
0.159/0.153 m/s, and DESCEND slow fractions increase from 11.2/8.1% to
13.9/15.1%. Endpoint/constraint-related hesitations still need work. Station 1
takes 3 seconds longer; no blanket speedup or collision-free claim is made.

Validation: recovery contracts 6/6, factory 21/21, walk 5/5; actual final
smooth missions PASS 2/2. Offscreen viewer default was also exercised through
2500 ticks (`continuous_viewer_{1800,2100,2500}.png`), not a human GUI check.

```bash
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1 --recovery-motion direct
OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_factory_recovery.py \
  --station 1 --motion-profile smooth --out results/factory/smooth_check.json
```

## Current result

The optional recovery controller physically walks to the dropped part, stops,
grasps bilaterally and lifts it in the existing factory. It does not teleport
G1, reset into a standalone grasp scene, weld the block, or complete the repair.

| dropped_part / default layout | Station 0 | Station 1 |
| --- | --- | --- |
| Functional lift | PASS | PASS |
| Maximum vertical bottom clearance | 67.69 mm | 59.96 mm |
| Continuous airborne bilateral support | 5.00 s | 5.00 s |
| Episode steps (10 ms each) | 5359 | 5672 |
| Largest base displacement per step | 5.69 mm | 6.18 mm |
| Fell | no | no |
| Maximum sampled hand/object penetration | 4.88 mm | 5.02 mm |
| Ticks with factory forbidden-body contact | 838 | 823 |
| Walking/rail contacts | none | none |

The table is the final raised-hand walking bias -0.08, seed 0, both stations.
Local evidence: `results/factory/recovery_brake_st{0,1}.json`. The previous +0.1
bias also lifted successfully and was repeated with seeds 0/1, but hit the rail
during walking. With the station overridden those seeds produced the same
fault timing/scene: repeatability, **not four generalization conditions**.

**The collision-free mission is NOT passed.** Functional lift is separate from
the existing forbidden-body contact flag. The evaluator reports both, including
pair-level force/penetration samples. Contact quality needs work before approving
these episodes for imitation learning. Place/verification/restart is not built;
the task remains RECOVERING after lifting. The misplaced scenario is not verified
for this new controller. Default `--walk-to` navigation is unchanged.

## Approach posture: direct instead of a clearance spread (2026-09-11)

The baseline reached the block through `ARM_LATERAL_CLEARANCE`, which drives the
palms **920 mm apart** (up from 470 mm standing) to grasp a **120 mm** block, and
then `FOREARM_FORWARD_REACH`, which swings that width back in. The factory
pipeline paid for it **twice**: once before walking, and again because the GRASP
phase constructed a fresh expert that restarted from `STABLE_START`. Measured at
station 0, arrival at step 1188 was followed by 412 ticks of re-preparation
before the object-relative align even began.

`--recovery-motion direct` enters `WRIST_SIDE_GRASP_ALIGN` straight from the
pose walking left the arms in. That state was already the right machinery: it
reads the **measured** palm poses as its start, targets an object-relative
side-grasp pose, and waypoints position and orientation together while ramping
finger curl. Its own docstring records that reorienting at a fixed position
self-collides while reorienting *during* translation does not. The preshape is
opened up front so the hand shapes during the reach rather than in a separate
stop.

| station 1 layout, seed 0 | baseline st0 | direct st0 | baseline st1 | direct st1 |
| --- | --- | --- | --- | --- |
| Functional lift | PASS | PASS | PASS | PASS |
| Episode steps | 5359 | **5179** | 5672 | **5292** |
| Max vertical clearance | 67.69 mm | **79.92 mm** | 59.96 mm | **73.08 mm** |
| Airborne bilateral support | 5.00 s | 5.00 s | 5.00 s | 5.00 s |
| Clearance-spread entries | 2 | **1** | 2 | **1** |
| Arrival to first contact | 2383 | 2357 | 2508 | 2357 |
| Max hand/object penetration | 4.88 mm | 5.31 mm | 5.02 mm | 6.19 mm |
| Forbidden-body contact ticks | 838 | 938 | 823 | 840 |
| Fell | no | no | no | no |

Lift height improved by 12-13 mm at both stations and the 5 s hold is unchanged
-- the success criterion was **not** relaxed to buy speed. Contact quality moved
the wrong way slightly (penetration +0.4/+1.2 mm, forbidden ticks +100 at
station 0); that is on top of an already-unapproved contact baseline, not a new
clean result.

### What it took to work on both stations

The first direct attempt lifted at station 0 but lost contact during the lift at
station 1. Measured cause, not guessed: the align finished **loose**, leaving
the palms 9 mm wider (±0.265 vs ±0.256 from the object) with the approach axes
splayed further out (0.471 vs 0.435). The direct align path is **0.362 m**
against the baseline's ~0.18 m, so the stock 14 waypoints made each
interpolation step twice as large. Scaling the waypoints with the measured path
length (to 28) and the align's step budget with it fixes both stations. The
align's own convergence test is unchanged.

`--recovery-motion baseline` still runs the original path and is kept as the
comparison point. `compact`, the earlier experiment that only skipped the
duplicated preparation, is also still selectable.

**The pre-walk spread is NOT removed.** `PREPARE_HANDS` still opens the arms
before walking, and the walking gait bias was tuned for that raised-hand pose,
so narrowing it needs walking re-verification. Only the second, post-arrival
spread is gone.

## Run

```bash
# Policy asset, if not already installed:
python3 scripts/install_g1_walk_policy.py

DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 0
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1

OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_factory_recovery.py \
  --station 0 --out results/factory/check_st0.json --render
OPENBLAS_NUM_THREADS=1 python3 scripts/test_factory_recovery.py
```

The viewer pauses on LIFTED/FAILED to preserve the actual end state. `R` resets
and starts a new attempt; `1`/`2` select station cameras. Manual A/C task signals
are disabled in automatic recovery. `--recover` and `--walk-to` are mutually
exclusive. Offscreen runs require a sufficient explicit step budget, for example
`--recover --offscreen --steps 6500 --capture 400 2100 4500 6500`.

The evaluator exits nonzero for functional failure. A zero exit code does not
certify collision-free recovery; check `collision_free_lift_success` and contact
metrics. The fast test script checks construction/stepping/reset contracts only.

Station-0 pair audit (previous +0.1 bias): during WALK, torso/near rail
peaked at 202.3 N with 0.456 mm penetration (16 sampled contacts). During GRASP,
torso/right thumb peaked at 9.27 N/1.32 mm; torso/shoulder-yaw contacts peaked at
7.64 N (right) and 10.12 N (left), with at most 0.157 mm penetration. These are
real contacts, not a false detector flag. No collision filters were disabled.
See `results/factory/recovery_contact_audit.json` for counts and phase labels.
The final -0.08 bias removes that walking/rail contact on both stations. Remaining
GRASP contacts: station 0 shoulder/torso peak 7.88 N, station 1 thumb/torso peak
30.96 N; maximum associated penetration 0.105/1.09 mm. Do not call it collision-free.

## What changed and why

1. **One physics world.** `SharpaGraspEnv` can be a view of the factory's exact
   model/data, with named part joint/body/geom and support geom. Its reset is
   rejected in this mode. The supervisor advances the scripted factory and
   grasp physics once per tick, never both env stepping loops.
2. **Prepare before the rail.** Raising hands after arriving can sweep thumbs
   through the conveyor rail and topple G1. Hands are prepared in free space
   after the fault request, before walking toward the part.
3. **Actual-part navigation.** A separate integral controller targets the
   measured part minus a 0.27 m forward offset, slows lateral motion near it,
   then hands off in double support. A -0.08 forward command bias compensates
   the raised-hand gait's different operating point, removing the previous
   forward overshoot into the rail. This is not the nominal marker gate.
4. **Measured stance handoff.** Hold the measured leg targets, not the previous
   gait command. Preserve the crouched stance and capture neutral ankle targets;
   forcing all knees/ankles to zero was tested and rejected after falls.
5. **Acquire with fingers too.** The dropped cube is rotated. A mandatory
   palm-first latch prevented finger closure. The factory path allows finger
   acquisition without inventing a palm-contact flag.
6. **Finish settling before rejecting the grasp.** A 6 mm squeeze ramps over
   2 s; factory FORCE_SETTLE grants 2.5 s to acquire stable support. The old 0.5 s
   window could reject the grasp before the squeeze completed. Airborne contact
   loss still fails after 0.5 s. Standalone defaults remain palm-first, 3 mm,
   0.5 s, so its canonical rollout is unchanged.

Lift evidence is the oriented box's **bottom above the belt top**, not unsigned
distance after sliding off the belt. Both real hand forces must be present and
opposing, and no other body may support the object for five continuous seconds.

## Remaining work

- Eliminate the recorded forbidden-body contacts and reduce grasp penetration.
- Test misplaced parts, genuinely varied fault poses, and arrival disturbances.
- Add controlled place, physical verification and production restart.
- Define and replay the **complete** factory command stream before any BC data
  approval. The supervisor includes leg and auxiliary grasp commands beyond the
  standalone action vector. No BC/PPO training was started in this integration.
- Environment source changes invalidate old demo hashes. Keep old recordings;
  do not change hashes to bypass validation. Record a new version when ready.
