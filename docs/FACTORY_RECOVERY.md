# Factory fault-to-lift integration (2026-09-11)

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
