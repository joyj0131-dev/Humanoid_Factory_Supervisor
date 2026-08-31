# Git Workflow / Phase Boundaries

이 문서는 이 저장소의 Phase 경계를 **커밋 기준으로 정확히** 고정하고,
공식 브랜치·태그·worktree 레이아웃과 일반 작업 절차를 기록한다. 37차
세션(Git 구조 정리, controller 결과를 새로 만든 세션이 아님)에 처음
작성됐다.

## Phase 경계 표

| Phase | 시작 커밋 | 현재/종료 커밋 | 공식 브랜치 | 상태 | 목적 |
|---|---|---|---|---|---|
| Phase 3 | (누적) | `849e1b0` | `archive/phase3-complete` | 완료 | BC pipeline foundation 스냅샷 |
| Phase 4 (Dex3) | `b46fe3b` | `bce1dec` | `phase4/dex3-grasp` | 실패로 종료·대체 | Dex3-1(3-finger) fixed-base bimanual grasp 연구 최종 스냅샷 |
| Phase 4.5 (Sharpa Wave) | `d038c5b` | `69276b3`(현재 tip) | `phase4.5/sharpa-wave` | 활성, 미완료 | Sharpa Wave(5-finger) 전환 및 grasp 연구 |
| Phase 4.5 single-hand prototype | — | `a0d286d` | `experiment/phase4.5-sharpa-single-hand` | 진단용 보존 | 공식 목표 아님, self-collision-avoidance geometry 참고용 |
| Phase 4.5 dirty viewer/rectangular 실험 | `69276b3` 기반 | (uncommitted) | `wip/phase4.5-viewer-rectangular` | 로컬 미커밋 | 10×15×10cm 직육면체/뷰어 실험 격리 보존 |

**중요한 구분**: `69276b3`은 Phase 4.5의 시작점이 아니라 **현재 tip**이다.
Phase 4.5의 실제 시작점은 `d038c5b`(Sharpa Wave 모델 vendoring/감사가
시작된 커밋)이다. 이 둘을 혼동하지 않는다.

## Annotated Tag

| Tag | 대상 커밋 | 의미 |
|---|---|---|
| `phase3-complete` | `849e1b0` | Phase 3 완료 |
| `phase4-start` | `b46fe3b` | Phase 4 시작 (origin/main tip) |
| `phase4-dex3-end` | `bce1dec` | Dex3 연구 종료 — **Gate A 실패**로 종료·대체 |
| `phase4.5-sharpa-start` | `d038c5b` | **Phase 4.5 실제 시작점** — Sharpa Wave vendoring/감사 |
| `phase4.5-sharpa-integration` | `540c9a4` | Stage 2 — G1+Sharpa 통합 완료 |
| `phase4.5-single-hand-prototype` | `a0d286d` | 진단용 단일손 prototype, 공식 목표 아님 |
| `phase4.5-bimanual-prototype` | `69276b3` | 현재 tip, **Gate A 미통과** |

## Ancestry 검증 (37차 세션에 실측)

```
git merge-base --is-ancestor bce1dec d038c5b   # YES — bce1dec가 d038c5b의 직접 선행 계열
git merge-base --is-ancestor d038c5b phase4.5/sharpa-wave  # YES
git rev-list --count bce1dec..phase4.5/sharpa-wave   # 5
git log --oneline bce1dec..phase4.5/sharpa-wave
#   69276b3 fix(grasp): restore bimanual Sharpa grasp and Gate A semantics
#   a0d286d feat(grasp): add Sharpa multi-finger grasp controller
#   540c9a4 feat(grasp): integrate Sharpa Wave hands with Unitree G1
#   54728b4 docs(grasp): switch Phase 4 target end effector to Sharpa Wave
#   d038c5b feat(grasp): vendor and audit Sharpa Wave hand model
```

## Worktree 레이아웃

- **원본 workspace**: `/home/youngjin/Mujoco_humanoid` — 브랜치
  `wip/phase4.5-viewer-rectangular`. 10×15×10cm 직육면체/`--no-restart`/
  force 진단 + Sharpa viewer 실험이 **커밋되지 않은 상태로** 여기에만
  존재한다. `PROJECT_CONTEXT.md`/`CLAUDE.md`/`.claude`/`docs/history`
  같은 ignored/local-only 파일의 실제 원본이 위치하는 곳이기도 하다.
- **Clean Phase 4.5 worktree**: `/home/youngjin/Mujoco_humanoid_worktrees/phase4_5_sharpa`
  — 브랜치 `phase4.5/sharpa-wave`, `69276b3` checkout. 위 ignored/
  local-only 경로들은 원본 workspace로부터 심볼릭 링크했다(tracked
  파일/디렉터리는 절대 덮어쓰지 않음 — 링크 전 `git ls-files`로 확인).
  Sharpa 대용량 mesh(`assets/robots/sharpa_wave/{left,right}_sharpa_wave/`)와
  `assets/robots/g1/`도 동일하게 링크했다(둘 다 `.gitignore` 대상이라
  clean checkout에는 애초에 존재하지 않음).

## 일반 작업 절차

1. **공식 Sharpa 개발은 clean `phase4.5/sharpa-wave` worktree에서
   수행한다.** 이 브랜치의 커밋만 신뢰할 수 있는 재현 가능한 상태다.
2. rectangular/no-restart/viewer 혼합 실험처럼 여러 목적이 한 파일에
   섞인 로컬 실험은 `wip/phase4.5-viewer-rectangular`(원본 workspace)
   에서만 보존한다.
3. WIP 브랜치의 변경을 clean 브랜치로 **통째로 복사(cherry-pick,
   diff 적용 등)하지 않는다.** 필요한 기능은 clean 브랜치에서 독립적으로
   재구현하고, 그 브랜치 자체의 테스트로 검증한 뒤 커밋한다 — 이렇게
   해야 두 브랜치가 서로 다른 실험을 안전하게 뒤섞지 않는다.
4. Gate 실패 상태를 성공으로 태그하거나 문서화하지 않는다("Stage/Phase
   완료"는 해당 acceptance criteria를 전부 만족했을 때만 사용).
5. **Phase 5(whole-body 이동, IL/BC/PPO 등)는 Phase 4.5의 Gate
   A→B→C→D를 전부 통과하기 전까지 시작하지 않는다.**
6. 공식 브랜치에 병합하기 전 반드시: 회귀 테스트 실행 → 결과 보고 →
   실패 시 수정 → 성공 확인 후에만 다음 단계 진행 (project skill의
   Verification rule과 동일).

## 다음 세션 시작 전 확인할 빌드 전제조건

`phase4.5/sharpa-wave`를 clean checkout하면 `model_builder.
build_grasp_model_sharpa()`가 참조하는 `GraspEnvConfig.
effective_object_half_extents`가 아직 커밋되어 있지 않아
`SharpaGraspEnv` 생성이 즉시 `AttributeError`로 실패한다(37차 세션에
`git log -S`로 확인: 이 의존성은 540c9a4부터 이미 존재). 다음 세션은
이 속성을 Sharpa 트랙에 필요한 최소 형태로 정식 커밋하는 것부터
시작해야 한다 — 이는 controller/IK/Gate 기준 변경이 아니라 순수 빌드
의존성 정리다. 상세: `docs/END_EFFECTOR_SHARPA_WAVE.md`.
