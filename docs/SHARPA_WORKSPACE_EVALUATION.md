# Sharpa grasp workspace evaluation — 2026-09-05

## Result and scope

Baseline `0e0a314` controller: 2/14 successful physical grasp/lift rollouts.
Updated controller: 14/14 evaluation cases and 4/4 separate combined cases.
The same controller/configuration is used for every final case, without
case-specific tuning, welded objects, pose teleportation or geometry changes.

Success requires actual bilateral hand/wrist support, at least 50mm signed
object/table clearance and 5 continuous seconds of airborne hold. The final
evaluator also rejects table support during AIR_HOLD. Historical thumb-specific
Gate A is separate; these results do not declare that Gate or Phase 4.5 complete.

All objects weigh 0.1kg with friction `(1.5, 0.01, 0.0001)`. Shape experiments
keep mass fixed, not density. Gravity compensation and the existing post-contact
`noslip_iterations=10` setting are unchanged. This is fixed-base manipulation.

| Evaluation cases | Conditions | Result |
| --- | --- | --- |
| Position, 12cm cube | x=0.260/0.265/0.270/0.275/0.280m at y=0; y=±0.005/±0.010m at x=0.270 | 9/9 |
| Other cubes | 11cm and 13cm at x=0.270, y=0 | 2/2 |
| Rectangular box | full XYZ size 0.10×0.15×0.10m | 1/1 |
| Rotation, 12cm cube | yaw ±5° at x=0.270, y=0 | 2/2 |

Separate combinations were specified before tuning and not used to select
controller settings:

| x (m) | y (m) | yaw | cube size | Result |
| --- | --- | --- | --- | --- |
| 0.267 | -0.003 | -2° | 12cm | PASS |
| 0.273 | +0.003 | +2° | 12cm | PASS |
| 0.266 | +0.004 | 0° | 11.5cm | PASS |
| 0.274 | -0.004 | 0° | 12.5cm | PASS |

These are finite scene checks, not a statistical success-rate estimate or proof
that every point/combination within these limits works. Mass/friction changes,
larger offsets, disturbances, irregular objects and perception errors remain
unvalidated. Different seeds without scene randomization are only repeats.

Across the final 18 runs, final clearance is 68.37–84.35mm and the minimum
continuous AIR_HOLD clearance is 64.72mm. Airborne table force is 0N throughout.
Hold-phase peak torso/arm contact is 2.11N; maximum object-contact penetration is
9.91mm (13cm cube). The canonical cube remains at 4.05mm maximum penetration.
These are soft-contact simulation results, not penetration-free precision grasps.
In particular, do not automatically accept the 13cm cube's trajectory as a clean
training demonstration merely because the lift succeeded. Contact quality needs
separate review/improvement before large-scale dataset collection.

Canonical additional verification: 5 more seconds of airborne hold after SUCCESS,
then actuator-driven opening/separation returns the free object to the table.
Offscreen front/top images were generated; the success image was visually
inspected. No live GUI window was opened for this verification.

## Actual fixes

1. FORWARD_REACH previously stopped solving after its last waypoint, with the
   IK tolerance exactly at the physical tracking gate boundary. Final-target
   correction now warm-starts from actual qpos after the nominal settling period,
   using 1mm IK tolerance while keeping the physical 10mm/15-tick gate unchanged.
   Starting tighter correction immediately disturbed a formerly working approach;
   waiting until step 400 preserves the nominal successful transit.
2. The support census omitted G1 wrist contacts even though the grasp physically
   uses them. The wrist balances the fore/aft force of the fingers. Omitting it
   falsely reported CONTACT_LOST while contact existed. The census now includes
   wrist roll/pitch/yaw bodies but excludes forearms and torso. A controlled real
   collision test verifies the previously missing wrist force.
3. Smaller/rectangular objects could have opposing physical support yet wait
   forever for historical index/middle+wrap topology. After 3s acquisition, the
   enveloping path can progress with 0.3s continuous current bilateral opposing
   support plus actual non-thumb finger contact on both hands. Old touch flags
   alone cannot satisfy this fallback. Strict Gate A itself is unchanged.
4. Object dimensions, yaw and seeded XY/yaw reset variation are now explicit.
   Separation and side-face diagnostics use the object's actual local frame and
   XYZ extents, not a world-axis cubic approximation.

No extra squeeze-force tuning or model/actuator parameter changes were needed.
The existing 3mm squeeze is retained. A small-box height/width-offset experiment
did not solve acquisition and was not adopted.

## Reproduce

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_workspace.py
OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_sharpa_workspace.py --suite all --workers 3 --output results/sharpa_workspace/final.jsonl
OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_sharpa_workspace.py --suite heldout --workers 2 --output results/sharpa_workspace/final_heldout.jsonl
OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_grasp_lift.py --seeds 0
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --object-size 0.10 0.15 0.10 --no-restart
```

`--object-size` means full XYZ dimensions in metres; `--object-half-size` remains
available for cubic half-extents. `--object-pos-y` and `--object-yaw-deg` expose
the same scene settings in the viewer. Configuration/reset APIs use radians.
Sizes are set when building an environment (correct mass/inertia); XY/yaw can
be overridden with `reset(options={"object_xy": [...], "object_yaw_rad": ...})`.

JSONL includes successes and failures, transitions, support-force traces, actual
clearance, air/table force, hold penetration/torso force, configuration, HEAD,
dirty status and source hashes. Results/images are ignored; this summary and
evaluation source are tracked. Earlier exploratory JSONL files predate the
extra provenance/force fields. No camera/vision estimate is used in these runs.

## Before imitation learning

Regression checks: Foundation env/expert/BC/whole-body and Sharpa model/integration/
hand-demo suites 92/92; workspace tests 4/4; selected bimanual action/observation/
strict-Gate contracts 8/8. The full historical bimanual suite, containing obsolete
failure/pose assertions, was not claimed clean or rerun in full. The physical
18-scene benchmark and canonical extended-hold/release test are separate checks.

**Done, 2026-09-06**: the replayable command interface now exists and a
10-scene pilot has been recorded and replayed. `SharpaGraspCommand` carries the
25-D action together with the 16 preshape actuator targets (including the thumb
CMC_FE nudge) and `noslip_iterations`, which the expert previously wrote outside
its returned action. All 10 recordings replay from commands alone, with no expert
constructed, to a maximum error of exactly 0.0 across qpos/qvel/ctrl/targets/
observation/telemetry/time. Stripping the preshape commands from a copy of a
successful recording is detected as a replay mismatch -- note that the stripped
copy still lifts the block, so a success-only check would not have caught it.
See the README's recording/replay section.

Still open before training: the recorded 129-D observation does not determine the
recorded command, because the expert is a state machine using contact information
that is not in the observation. Shape context must be available to the learner,
not only the expert. Recordings are marked `learner_ready=False` and
`quality_review_required=True`; replay fidelity is not demonstration quality.
Existing 43-D/14-D Foundation learning contracts remain unchanged.

Camera-derived object pose can later replace simulator state through a separate
perception interface. The present work neither implements vision nor proves
robustness to camera occlusion, calibration error or real-world dynamics.
