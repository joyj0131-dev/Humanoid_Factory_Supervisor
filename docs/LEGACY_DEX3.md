# Legacy Dex3 Track

Dex3 연구는 실패를 숨기거나 삭제한 것이 아니라 별도 역사적 baseline으로
보존한다.

## Canonical preservation point

- Branch: `phase4/dex3-grasp`
- Commit: `bce1dec`
- Tag: `phase4-dex3-end`
- Outcome: Gate A 실패 상태로 종료·Sharpa Wave로 대체

활성 Sharpa 브랜치에서는 아래 source가 제거됐다. 위 branch/tag는
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

## Active branch status

- 일반 timing utility는 `expert/timing.py`로 이동했다.
- Foundation과 whole-body 내부 모델도 bare G1+Sharpa로 이전했다.
- viewer의 hand-model dispatch와 legacy diagnostic mode를 제거했다.
- `coupled_ik.py`와 `pose_ik.py`는 현재 Sharpa controller가 직접 사용하는
  일반 G1 arm solver이므로 유지한다.
- 로컬 `assets/robots/g1/`는 upstream menagerie bundle 전체가 Git ignore된
  설치 자산이다. active code는 그중 `g1.xml`만 읽으며
  `g1_with_hands.xml`을 참조하지 않는다.

Dex3와 Sharpa의 최종 비교가 필요하면 archive branch의 고정 결과를
사용한다. Gate 기준을 바꾸거나 서로 다른 task를 같은 비교로 제시하지
않는다.
