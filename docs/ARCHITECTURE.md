# Current Architecture

활성 브랜치의 robot model은 하나뿐이다: **bare Unitree G1 + bilateral
Sharpa Wave**. 손 종류를 선택하는 runtime 분기는 없다.

## Runtime flow

```text
scripts/view_whole_body.py
  ├─ --stand / --posture
  │    └─ WholeBodyEnv (37 action = legs12 + waist3 + arms14 + hands8)
  ├─ --planar
  │    └─ PlanarDebugEnv
  ├─ --grasp
  │    └─ SharpaGraspEnv (25 action, 129 observation)
  │         └─ SharpaBimanualGraspExpert
  │              ├─ CoupledBilateralIK
  │              ├─ pose_ik
  │              └─ timing.sim_time_to_steps
  └─ --sharpa-hand-demo
       └─ SharpaHandDemo + SharpaGraspEnv
```

`model_builder.py`는 bare `assets/robots/g1/g1.xml`에 Sharpa를
programmatic하게 부착한다. 기존 keyframe 배열 순서에 의존하지 않고
joint/actuator 이름으로 stand keyframe을 재구성한다. 이 과정에서 과거
pre-hand-equipped base가 남기던 손목당 0.202839kg의 ghost hand mass도
제거됐다.

## Public contracts

- Foundation reach: observation 43, action 14. 기존 demonstration,
  checkpoint, Expert/BC/Evaluation 의미를 유지한다.
- Whole-body: action 37. 손 부분은 left/right 각각
  thumb/index/middle/wrap 네 그룹이다.
- Fixed-base Sharpa grasp: action 25, observation 129.
- G1+Sharpa compiled integration: `nq=80`, `nv=79`, `nu=73`.

## Source ownership

- `envs/task_config.py`, `whole_body_config.py`: G1 공통 설정
- `envs/model_builder.py`: G1 scene + Sharpa attachment
- `envs/sharpa_config.py`: Sharpa naming, joint roles, preshape
- `envs/sharpa_grasp_env.py`: fixed-base grasp physics/contact safety
- `expert/sharpa_bimanual_grasp_expert.py`: 공식 양손 controller
- `expert/sharpa_hand_demo.py`: free-space hand diagnostic
- `expert/coupled_ik.py`, `pose_ik.py`, `timing.py`: 공통 expert 도구
- `data/`, `imitation/`, `evaluation/`: Phase 2/3 및 이후 학습 기반

과거 Dex3 runtime/planner/test는 활성 branch에서 제거됐다. 재현이 필요하면
`phase4/dex3-grasp` branch 또는 `phase4-dex3-end` tag를 사용한다.

## Current blocker

Phase 4.5는 미완료다. `FOREARM_FORWARD_REACH` Gate(9.96mm)와 신규
Side-Grasp Posture Gate(양손이 물체 좌우에서 마주보고 손가락이 아래를
향하는 bilateral side-grasp 자세)는 모두 PASS했다. official rollout은
collision 없이 `FOREARM_SIDE_DESCEND`까지 진행하지만 그 상태의 후반
waypoint에서 hand-table collision(최대 약 18.21N, 8N 한계 초과)으로
막힌다. Gate를 완화하지 않고 이 충돌을 해결한 뒤 FINGERTIP_PRECONTACT/
contact로 진행한다.
