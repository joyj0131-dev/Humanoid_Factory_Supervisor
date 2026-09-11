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

scripts/view_factory.py
  └─ FactoryEnv (37 action, 122 observation, floating base)
       ├─ factory_model.build_factory_model (G1 + conveyor + 2 stations)
       ├─ ScriptedArm x2 (4-DoF + two-jaw gripper)
       ├─ NavigationTracker (nominal station gate)
       ├─ --walk-to: G1WalkPolicy + WalkToPose (existing navigation)
       └─ --recover: FactoryRecovery
            ├─ fault -> prepare hands clear of the rail
            ├─ G1WalkPolicy + actual-part PrecisionApproach
            ├─ measured leg-target handoff + ankle feedback
            └─ shared-model/data SharpaGraspEnv -> grasp expert -> physical lift
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
- Demo replay(schema 1): `SharpaGraspCommand` = action 25 + preshape 목표 16 +
  `noslip_iterations`. 이는 기존 action/observation 계약을 바꾸지 않고,
  action 밖에서 나가던 보조 명령을 명시적으로 포함시킨 것이다.
- Factory supervisor: action 37, observation 122 = WholeBodyEnv 83 + factory 39.
  compiled `nq=106`, `nv=103`, `nu=85`(G1 73 + 자동화 팔/그리퍼 12).
- Recovery supervisor 내부에서는 다리 목표와 grasp 25-dim/부가 손 명령을 함께
  구동한다. 이 복합 명령의 공장 데모 기록·재생/학습 인터페이스는 아직 미검증이다.

## Source ownership

- `envs/task_config.py`, `whole_body_config.py`: G1 공통 설정
- `envs/model_builder.py`: G1 scene + Sharpa attachment
- `envs/sharpa_config.py`: Sharpa naming, joint roles, preshape
- `envs/sharpa_grasp_env.py`: fixed-base grasp physics/contact safety,
  `capture_command()` / `step_command()`
- `envs/sharpa_command.py`: 검증되는 불변 완전 명령 dataclass
- `envs/factory_config.py`: workcell layout, 자동화 팔 spec, fault, Navigation Gate
- `envs/factory_model.py`: 2개 workcell scene 컴파일
- `envs/factory_env.py`: FactoryEnv, ScriptedArm, NavigationTracker
- `envs/whole_body_env.py`: `_build_model()` hook으로 FactoryEnv가 동일한
  37차원 action 규약과 step 의미를 재구현 없이 공유한다
- `data/sharpa_demo.py`: 기록/재생과 Expert flag를 보지 않는 `PhysicalMonitor`
- `expert/sharpa_bimanual_grasp_expert.py`: 공식 양손 controller
- `expert/sharpa_contact_lift.py`: 실제 양손 지지 및 물체-table 간격 기반 hold/lift
- `expert/factory_recovery.py`: 보행/파지를 한 physics step으로 연결.
  공유 grasp view는 `reset()` 금지이며 공장과 같은 model/data를 사용한다.
  성공은 물체 바닥의 벨트 위 수직 높이 + 실제 양손 지지 + 다른 지지물 접촉 없음으로 판정한다.
- `expert/sharpa_hand_demo.py`: free-space hand diagnostic
- `expert/coupled_ik.py`, `pose_ik.py`, `timing.py`: 공통 expert 도구
- `data/`, `imitation/`, `evaluation/`: Phase 2/3 및 이후 학습 기반
- `scripts/sharpa_demos.py`: 파일럿 기록/재생 CLI
- `scripts/test_sharpa_demo.py`: 명령 완전성·진단 격리·재생 부정 테스트
- `scripts/view_factory.py`, `scripts/test_factory.py`: 공장 viewer/테스트

과거 Dex3 runtime/planner/test는 활성 branch에서 제거됐다. 재현이 필요하면
`phase4/dex3-grasp` branch 또는 `phase4-dex3-end` tag를 사용한다.

## Current blocker

2026-09-11: dropped_part seed 0 양쪽 스테이션에서 live walk-to-lift 성공.
Place/검증/재가동 연결, 위치 불량 시나리오와 더 넓은 시작조건, 파지 접촉 품질,
공장 mission용 완전 명령 기록·재생은 별도 남은 작업이다. 아래는 단독 파지 트랙 기록이다.

2026-09-05 작업영역 확장: 같은 설정으로 위치·크기·yaw 14조건과 별도 조합
4조건의 물리 파지/상승이 성공했다. 최종 접근 재보정, 손목을 포함한 실제
지지력 집계, 작은 물체용 현재 접촉 기반 enveloping 전이를 추가했다.
환경은 XYZ 크기와 seeded XY/yaw reset을 지원한다. 자세한 검증 범위와
다음 IL 기록/재생 과제는 [작업영역 평가](SHARPA_WORKSPACE_EVALUATION.md)에 있다.
아래 기본 장면의 과거 수치는 다양한 조건 전체의 품질 보장이 아니다.

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
접촉 지점을 물체 로컬 좌표로 보면 양손 모두 서로 반대쪽 Y면(측면)을 누르는
구조라 실제로는 양손이 물체를 사이에 끼우는 협동 grasp에 가깝고, 손 안에서
엄지가 index/middle에 대립하는 tripod 구조는 아니다(엄지는 접촉력이 0에
가깝다). 사용자 지적대로 이건 정상이다 -- 이 프로젝트의 기본 설계가 애초에
양손 협동이지 한 손 tripod가 아니다.

[Precision-grasp session] 원래 `hold_squeeze_m=0.012`에서는 손바닥+손목이
지지력의 84~86%, 손가락은 14~16%뿐이었다. 손가락 쪽에 추가로 닫는 힘을
얹는 두 번의 시도(THUMB_OPPOSE 시간 연장, 이미 닿은 그룹만 살짝 더 조이기)는
모두 이미 성공하던 rollout을 CONTACT_LOST로 회귀시켰다 -- 12mm squeeze +
얼린 손가락 자세가 아주 미세하게 맞춰진 균형점이라 어떤 추가 힘도 이를
깨뜨렸다. 대신 `hold_squeeze_m` 자체를 실측 스윕한 결과 0.002m까지는
성공하고(0.001m는 실패) 원래 0.012m는 필요 이상으로 컸다는 게 드러났다.
`hold_squeeze_m=0.003`(실패 지점 대비 3배 여유)으로 낮추기만 해도 새 손가락
로직 없이 손가락 비중이 seed0/1/2에서 약 30~38%까지 오른다 -- 팔이 덜
누르는 만큼 이미 있던 손가락 접촉이 상대적으로 더 많은 일을 하게 된다.

## Demonstration recording and Expert-free replay

`SharpaGraspEnv.capture_command()`는 `expert.step()`이 반환한 25차원 action에
더해 Expert가 action 밖에서 직접 쓴 명령 -- preshape 16개 actuator 목표(엄지
CMC_FE nudge 포함)와 solver의 `noslip_iterations` -- 를 함께 포착한다.
`step_command()`는 그 보조 명령을 적용한 뒤 **기존 `env.step`을 그대로 호출**하므로
중력 보상과 substep force safety가 우회되지 않는다. 이 두 메서드는 기존
action_space 25 / observation_space 129를 바꾸지 않는다.

재생(`replay_episode`)은 grasp policy를 import하지 않고 IK를 다시 풀지 않으며,
기록된 qpos/qvel을 물리에 대입하지 않는다. 같은 seed로 reset한 뒤 저장된 명령만
실행하고 기록된 상태와 비교한다(허용 오차 1e-6, reset 시점은 1e-9). MuJoCo 버전,
`envs/*.py` hash, 컴파일 모델 hash, preshape actuator 순서, 배열 정렬, 기록된
step 수가 하나라도 어긋나면 재생 자체를 거부한다.

정렬 규약: `observations[t]`가 `commands[t]`보다 앞선다. 명령은 T개,
비교용 상태는 T+1개다.

`_empirical_group_closure_world` 진단은 이제 `copy.copy(env)` + 별도 `MjData` +
`mj_copyData`로 만든 **복제 환경**에서 실제 물리를 돌린다. 이전에는 실제 env에서
step한 뒤 qpos/qvel/ctrl만 되돌려서 시간, warm-start, controller target, counter가
남을 수 있었다. 복제본은 model을 공유하지만 `env.step`은 model을 쓰지 않는다.

기록 파일에는 `learner_ready=False`, `quality_review_required=True`가 들어 있다.
재생 성공과 학습 데모 품질 승인은 분리한다.

## Factory environment (2026-09-10)

두 workcell은 canonical grasp 관계(테이블 0.30m 앞, 부품 0.27m)를 rigid
transform한 것이며, 자기 좌표계에서 서로 완전히 동일하다(round-trip 1e-12,
base frame 재현 1e-6). manipulation pose 간격 2.868m.

자동화 팔은 실제 joint/position actuator로 구동한다. cycle은 **tip 좌표로
작성**하고 2-link closed-form IK로 관절각을 만든다 — 처음에 관절각을 직접
하드코딩했다가 tip이 테이블을 0.43m 관통한 실패를 겪었기 때문이다. 현재 cycle의
최소 테이블 여유는 +0.070m이고, 더 빡빡한 제약인 부품과의 최소 거리는 +48mm다
(0.82/0.88 tip에서는 각각 98/247회 접촉과 62/136mm 부품 이동이 발생했다 —
부품 윗면이 0.872m라는 것을 실측하고서야 잡혔다). 팔은 pick/place를 흉내만 내며
부품을 실제로 옮기지 않고, 건드려서도 안 된다.

Fault는 seed만으로 결정되며, 해당 팔은 pick waypoint에서 정지하고 부품은 drop
zone 0.05m 위에서 released되어 중력으로 낙하·정착한다(실측 rest z 0.8097m,
|qvel|max 0.00000, 관통 0.34mm). 반대쪽 셀은 계속 돌아간다(정상 0.830rad vs
정지 0.007rad).

**보행 제어기는 없다.** base는 free로 유지되고 NavigationTracker는 외부에서 준
궤적을 채점만 한다. step당 base 이동 0.05m 초과는 teleport로 실격 처리하므로
kinematic oracle이 보행으로 둔갑할 수 없다.

미해결 두 가지(실측):
- grasp Expert의 접근 offset이 world 축 기반이라 35° 셀에서 최대 169mm 어긋난다.
  또한 `object` 이름으로 물체를 찾는데 factory 모델에는 `wc0_part`/`wc1_part`뿐이다.
- Navigation Gate 허용 오차 0.10m와 검증된 grasp 범위 ±0.010m 사이에 10배 격차가
  있다. Gate를 통과해도 현재 grasp이 성공한다는 뜻이 아니다.
