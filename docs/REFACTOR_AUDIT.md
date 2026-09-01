# Phase 4.5 Refactor Audit

Audit 기준: `phase4.5/sharpa-wave` @ `999526d`.

분류는 tracked file 목록, Python AST import graph, `rg` symbol reference,
viewer/CLI entry point, 직접 실행 테스트를 함께 사용했다. 단순히 inbound
import가 없다는 이유만으로 entry-point script를 미사용으로 판정하지
않았다.

## Classification

| 분류 | 파일/디렉터리 | 판정 |
|---|---|---|
| Sharpa 실행 필수 | `envs/sharpa_config.py`, `sharpa_grasp_env.py`, `expert/sharpa_bimanual_grasp_expert.py`, `model_builder.py`, `grasp_config.py`, `coupled_ik.py`, `pose_ik.py`, `scripts/view_whole_body.py`, Sharpa binary assets | 유지 |
| Sharpa 진단·재현 | `expert/sharpa_hand_demo.py`, `scripts/audit_sharpa_wave.py`, `audit_sharpa_mount.py`, `diagnose_precontact_gap.py`, `diagnose_clearance_wrist_spike.py`, `install_sharpa_wave_assets.py`, `test_sharpa_*` | 유지 가치 있음 |
| Sharpa 단일손 진단 | `expert/sharpa_grasp_expert.py`, `scripts/test_sharpa_single_hand_diagnostic.py` | 공식 controller 아님; archive 후보지만 현재 참고 가치 있음 |
| G1 공통 기반 | `task_config.py`, `whole_body_config.py`, `frames.py`, `reward.py`, `humanoid_reach_env.py`, `whole_body_env.py`, `planar_debug_env.py`, `model_builder.py`, `hand_synergy.py`, `ik_solver.py`, `pose_ik.py`, `coupled_ik.py`, `scripted_expert.py` | 유지 |
| Phase 5+ 학습 기반 | `data/`, `imitation/`, `evaluation/`, `configs/`, `collect_demos.py`, `train_bc.py`, `evaluate_bc.py`, Foundation tests | 유지 |
| Dex3 legacy runtime | `grasp_env.py`, Dex3 부분의 `grasp_config.py`/`model_builder.py`, `grasp_expert.py`, `test_grasp.py`, viewer의 Dex3 modes | 공통 추출 전 삭제 금지 |
| Dex3 legacy planners | `diagonal_feasibility.py`, `grasp_feasibility_map.py`, `grasp_wrench_diagnostics.py`, `manifold_grasp_planner.py`, `substep_safety_trace.py`, `thumb_contact_diagnostics.py`, `tripod_closure_planner.py`, `whole_body_diagonal_feasibility.py`와 대응 scripts | 활성 브랜치 제거 후보; archive에는 보존 |
| 프로젝트 인프라 | `requirements.txt`, `.gitignore`, tracked docs, Sharpa license/checksum metadata | 유지 |
| 생성 결과 | `results/grasp_feasibility_map/`, `results/*.json`, session image/video dirs | 이미 Git ignore; 소스 아님 |
| 학습 산출물 | `datasets/`, `checkpoints/` | Git ignore지만 Foundation 재현에 필요; 삭제 금지 |
| 캐시 | 모든 `__pycache__/`, `*.pyc`, `.pytest_cache/` | 즉시 삭제 가능, 자동 재생성 |
| 로컬 운영 문서 | `PROJECT_CONTEXT.md`, `CLAUDE.md`, `docs/history/` | Git ignore; 현재 상태/세션 원문 보존 |
| 로컬 백업 | `/home/youngjin/Mujoco_humanoid_local_backups/phase4_5_viewer_rectangular_69276b3.patch` | 저장소 밖 비상 복구용, 이번 정리 대상 아님 |
| 완전히 미사용 | **확정된 tracked source 없음** | 삭제 안 함 |

`results/bc/train_log.csv`와 `results/bc/eval_results.csv`는 tracked Phase 3
증거이므로 generated result라는 이유로 삭제하지 않는다.

## Verified dependency edges blocking Dex3 deletion

```text
SharpaGraspEnv
  -> grasp_config.GraspEnvConfig
  -> model_builder.build_grasp_model_sharpa
  -> task_config / whole_body_config / sharpa_config

SharpaBimanualGraspExpert
  -> coupled_ik.CoupledBilateralIK
  -> pose_ik
  -> grasp_expert.sim_time_to_steps

view_whole_body.py
  -> Sharpa runtime
  -> Dex3 runtime
  -> legacy diagonal/whole-body diagnostics
```

추가로 `SharpaGraspEnv`는 `hand_synergy`를 import하지만 실행 symbol을
사용하지 않는다. 이것은 다음 import-cleanup 후보이지 `hand_synergy.py`
삭제 근거가 아니다. Foundation whole-body 코드가 그 파일을 사용한다.

## Baseline behavior snapshot

리팩터링 전 직접 실행:

- `test_env.py`: 17/17
- `test_expert.py`: 14/14
- `test_bc.py`: 17/17
- `test_whole_body.py`: 21/21
- `test_sharpa_wave_model.py`: 9/9
- `test_sharpa_g1_integration.py`: 10/10
- `test_sharpa_hand_demo.py`: 4/4
- `test_sharpa_bimanual_grasp.py`: 14/15

마지막 1개 실패는 실제 SIZE_12 Gate A 성공을 아직 달성하지 못했다는
정직한 assertion이다. 공식 rollout은 deterministic하게 1070 step에서
`PRECONTACT_TRACKING_NOT_ACHIEVED`, bilateral streak `0/30`, 양손 접촉
0회로 종료했다. 통합 모델은 `nq=80`, `nv=79`, `nu=73`이다.

감사 후 추가 회귀도 기존 결과를 유지했다: Dex3 grasp 46/54, Sharpa
single-hand 7/7, substep/tripod/wrench/thumb diagnostics 각 7/7,
diagonal feasibility 13/14, whole-body diagonal 5/6. 실패 항목은 모두
기존의 정직한 Gate/feasibility 실패다. `test_manifold_grasp_planner.py`는
약 150초 동안 출력 없이 실행되어 중단했다. source 변경이 없는 이번
세션의 회귀로 판정하지 않지만, 별도의 테스트 실행시간 진단이 필요하다.

## Immediate cleanup decision

- 삭제 가능: exact `__pycache__` directories만 제거.
- 유지: ignored result는 작고 재현 근거이므로 이번 세션에는 보존.
- 보류: 모든 tracked source 이동/삭제. 현재 세션은 행동 불변 감사다.
- 다음 리팩터링 첫 단계: `sim_time_to_steps`와 공통 grasp timing/contact
  primitives를 작은 common module로 추출하고 두 controller test를 함께
  실행한다.

## Staged refactor plan

1. Common timing/contact utilities extraction.
2. Shared grasp scene config와 hand-specific config 분리.
3. Builder를 base scene / Dex3 / Sharpa attachment로 함수 단위 분리.
4. Viewer mode 구현 분리, CLI dispatch만 유지.
5. Sharpa import graph의 Dex3 edge 0 확인.
6. Dex3-only planner/diagnostic을 활성 브랜치에서 제거하고 archive branch
   재현성 확인.

각 단계는 단일 목적 커밋, 전후 모델 contract와 실패 결과 비교를 요구한다.
