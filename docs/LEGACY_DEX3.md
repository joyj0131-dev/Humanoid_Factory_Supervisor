# Legacy Dex3 Track

Dex3 연구는 실패를 숨기거나 삭제한 것이 아니라 별도 역사적 baseline으로
보존한다.

## Canonical preservation point

- Branch: `phase4/dex3-grasp`
- Commit: `bce1dec`
- Tag: `phase4-dex3-end`
- Outcome: Gate A 실패 상태로 종료·Sharpa Wave로 대체

활성 Sharpa 브랜치에서 Dex3 source를 정리하더라도 위 branch/tag는
변경하지 않는다.

## Dex3-only code

### Runtime

- `humanoid_learning/envs/grasp_env.py`
- `humanoid_learning/expert/grasp_expert.py`의 controller/contact 로직
- `scripts/test_grasp.py`
- viewer의 Dex3 `--grasp`, `--grasp-safety-latch`, diagonal modes

### Planners and diagnostics from sessions 1~34

- `expert/diagonal_feasibility.py`
- `expert/grasp_feasibility_map.py`
- `expert/grasp_wrench_diagnostics.py`
- `expert/manifold_grasp_planner.py`
- `expert/substep_safety_trace.py`
- `expert/thumb_contact_diagnostics.py`
- `expert/tripod_closure_planner.py`
- `expert/whole_body_diagonal_feasibility.py`
- 대응하는 `audit_*`, `diagnose_*`, `plan_*`, `run_*`, `test_*` scripts

## Not yet Dex3-only

다음 파일은 이름이나 역사 때문에 Dex3처럼 보여도 Sharpa가 현재 직접
사용하므로 삭제할 수 없다.

- `envs/grasp_config.py` — Sharpa env와 tests가 `GraspEnvConfig` 사용
- `envs/model_builder.py` — `build_grasp_model_sharpa`와 attachment 포함
- `expert/grasp_expert.py` — Sharpa가 `sim_time_to_steps` import
- `expert/coupled_ik.py`, `pose_ik.py` — 양 hand model 공통
- `envs/hand_synergy.py` — Foundation/whole-body 회귀가 사용; Sharpa env의
  import는 현재 불필요하지만 파일 자체는 legacy 전용이 아님

## Removal criteria on the active branch

1. 공통 유틸을 `common`으로 추출한다.
2. Sharpa import graph에서 Dex3 runtime/planner로 향하는 edge가 0인지
   AST와 `rg`로 확인한다.
3. Foundation, whole-body, Sharpa tests가 통과한다.
4. Dex3 archive branch/tag에서 legacy 실행이 재현 가능한지 확인한다.
5. 그 뒤 활성 브랜치에서 Dex3-only 파일을 제거한다.

Dex3와 Sharpa의 최종 비교가 필요하면 archive branch의 고정 결과를
사용한다. Gate 기준을 바꾸거나 서로 다른 task를 같은 비교로 제시하지
않는다.
