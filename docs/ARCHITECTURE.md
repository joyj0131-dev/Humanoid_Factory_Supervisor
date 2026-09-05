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
  │              ├─ SharpaContactLift (contact/hold/lift feedback)
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
- `expert/sharpa_contact_lift.py`: 실제 양손 지지 및 물체-table 간격 기반 hold/lift
- `expert/sharpa_hand_demo.py`: free-space hand diagnostic
- `expert/coupled_ik.py`, `pose_ik.py`, `timing.py`: 공통 expert 도구
- `data/`, `imitation/`, `evaluation/`: Phase 2/3 및 이후 학습 기반

과거 Dex3 runtime/planner/test는 활성 branch에서 제거됐다. 재현이 필요하면
`phase4/dex3-grasp` branch 또는 `phase4-dex3-end` tag를 사용한다.

## Current blocker

기본 rollout은 CONTACT_ACQUIRE → THUMB_OPPOSE → FORCE_SETTLE →
TABLETOP_HOLD → LIFT → AIR_HOLD → SUCCESS까지 진행한다. `physical_grasp_success`는
실제 양손 지지와 5cm 이상 table clearance를 연속 5초 유지해야 한다.
기존 엄지-specific Gate A는 별도 진단으로 남으며 아직 미통과다.
`contact_driven_lift=False`는 과거 모든 그룹 접촉 대기 경로를 비교할 때만 쓴다.

접촉 이후 `noslip_iterations=10`으로 수치적 creep를 줄이고, 달성한 손가락
목표는 유지한다. 각 그룹에 계속 더 닫는 명령을 적분하는 방식은 fingertip이
블록 모서리에서 굴러 벗어나는 문제가 있어 사용하지 않는다. 기존 substep
force safety는 유지한다. reset은 접근 시점 solver 설정을 복원한다.

기본 12cm/0.1kg fixed-base 장면의 파지·상승은 검증했지만 다양한 물체,
엄지 대향접촉, 초기 접근의 손목 transient 및 전체 Phase 4.5 승인은 별도다.
실제 지지는 손가락+손바닥+손목 접촉을 사용하며 hold 구간 최대 관통은
약 3.85mm다. 순수 fingertip grasp 또는 기존 1mm 관통 Gate 통과로 해석하지 않는다.
seed0 성공 rollout의 몸체별 접촉력 합산: 손바닥(hand_C_MC)+손목(wrist_yaw_link)이
지지력의 약 84~86%, 손가락(주로 wrap, middle 일부)은 약 14~16%다. 접촉 지점을
물체 로컬 좌표로 보면 양손 모두 서로 반대쪽 Y면(측면)을 누르는 구조라 실제로는
양손이 물체를 사이에 끼우는 협동 grasp에 가깝고, 손 안에서 엄지가 index/middle에
대립하는 tripod 구조는 아니다(엄지는 이 rollout 내내 접촉력이 0이다).
