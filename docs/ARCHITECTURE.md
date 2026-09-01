# Current Architecture

이 문서는 `phase4.5/sharpa-wave`의 **현재 실제 구조**를 설명한다. 아래의
목표 디렉터리 구조는 아직 구현되지 않았으며, 현재 파일 위치와 혼동하지
않는다.

## Runtime flow

```text
scripts/view_whole_body.py
  ├─ --grasp --hand-model sharpa
  │    ├─ GraspEnvConfig                 (envs/grasp_config.py, 현재 혼합)
  │    ├─ SharpaGraspEnv                 (envs/sharpa_grasp_env.py)
  │    │    └─ build_grasp_model_sharpa (envs/model_builder.py, 현재 혼합)
  │    └─ SharpaBimanualGraspExpert
  │         ├─ CoupledBilateralIK
  │         ├─ pose_ik
  │         └─ sim_time_to_steps         (expert/grasp_expert.py, Dex3 파일)
  ├─ --sharpa-hand-demo
  │    └─ SharpaHandDemo + SharpaGraspEnv
  └─ --grasp --hand-model dex3
       └─ FixedBaseGraspEnv + BimanualSidePinchExpert
```

AST import graph, CLI dispatch, 직접 실행 테스트를 함께 감사했다. Sharpa는
Dex3 controller를 상속하지 않지만 다음 결합은 아직 남아 있다.

- `SharpaGraspEnv`가 혼합 설정인 `GraspEnvConfig`와 혼합 builder인
  `model_builder`를 사용한다.
- Sharpa expert 두 개가 일반 시간 변환 함수 `sim_time_to_steps`를
  Dex3 전용 대형 파일 `grasp_expert.py`에서 import한다.
- `model_builder.py`가 Foundation, whole-body, Dex3, Sharpa builder를 모두
  담는다.
- `view_whole_body.py`가 모든 역사적 진단과 두 hand model dispatch를 모두
  담는다.
- `SharpaGraspEnv`의 `hand_synergy` import는 현재 실행 코드에서 사용되지
  않는다. 삭제는 다음 소규모 import 정리에서 테스트와 함께 수행한다.

따라서 지금 `grasp_expert.py`, `grasp_config.py`, `model_builder.py`를
Dex3 legacy라고 보고 삭제하면 Sharpa runtime도 깨진다.

## Current subsystems

### Foundation / Phase 5 이후 공통

- `envs/task_config.py`, `frames.py`, `reward.py`
- `envs/humanoid_reach_env.py`, `whole_body_config.py`,
  `whole_body_env.py`, `planar_debug_env.py`
- `expert/ik_solver.py`, `pose_ik.py`, `coupled_ik.py`,
  `scripted_expert.py`
- `data/`, `imitation/`, `evaluation/`
- `configs/`, `scripts/collect_demos.py`, `train_bc.py`, `evaluate_bc.py`

Foundation Phase 1/2와 50-demo dataset은 삭제·재작성하지 않는다.

### Active Sharpa track

- `envs/sharpa_config.py`, `sharpa_grasp_env.py`
- `expert/sharpa_bimanual_grasp_expert.py` — 공식 양손 controller
- `expert/sharpa_hand_demo.py` — free-space 진단
- `expert/sharpa_grasp_expert.py` — 단일손 exploratory diagnostic
- `assets/robots/sharpa_wave/` metadata와 설치 스크립트
- Sharpa audit/integration/controller tests

### Dex3 legacy track

Dex3 fixed-base env/controller와 1~34차 planner/diagnostic은 현재 브랜치에도
남아 있지만, 공식 보존점은 `phase4/dex3-grasp`의 `bce1dec`다. 구체적
범위와 제거 조건은 [LEGACY_DEX3.md](LEGACY_DEX3.md)를 따른다.

## Compiled model contracts

현재 G1+Sharpa 통합 테스트에서 확인한 값:

- `nq=80`, `nv=79`, `nu=73`
- Sharpa actuator 44개(손당 22개)
- Sharpa env action 25차원, observation 129차원
- 좌우 fingertip과 joint/actuator naming은 통합 테스트로 고정

리팩터링 단계에서는 이 값, 이름, joint range, actuator gain, collision,
mass/inertia가 모두 동일해야 한다.

## Target layout (incremental, not implemented yet)

```text
humanoid_learning/
  common/       G1 scene/model primitives, shared IK, timing/contact tools
  sharpa/       Sharpa attachment/config/env/expert
  legacy_dex3/  Dex3 env/controller and retained diagnostics
scripts/
  view_whole_body.py  thin hand-model/mode dispatch only
```

한 번에 이동하지 않는다. 권장 순서:

1. `sim_time_to_steps` 같은 무형태 공통 함수부터 `common`으로 이동한다.
2. 공통 grasp scene config와 hand-specific config를 분리한다.
3. `model_builder.py`의 scene base, Dex3 attachment, Sharpa attachment를
   함수 단위로 분리한다.
4. viewer mode 구현을 hand-specific 모듈로 옮기고 CLI만 남긴다.
5. 모든 import와 회귀 테스트가 안정된 뒤 Dex3 코드를 `legacy_dex3`로
   이동하거나 활성 브랜치에서 제거한다.

각 단계는 별도 커밋으로 만들고 전후 behavior snapshot을 비교한다.
