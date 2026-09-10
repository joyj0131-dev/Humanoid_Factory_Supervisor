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
| Phase 4.5 (Sharpa Wave) | `d038c5b` | `69276b3`(최신 controller checkpoint) | `phase4.5/sharpa-wave` | 활성, 미완료 | Sharpa Wave(5-finger) 전환 및 grasp 연구 |
| Phase 5 (Factory env) | `28c7364` | (진행 중) | `phase5/factory-environment` | 활성, 미완료 | 컨베이어 라인 + scripted 자동화 팔 2대 + 고장 주입 + 복구 신호 |
| Phase 4.5 single-hand prototype | — | `a0d286d` | `experiment/phase4.5-sharpa-single-hand` | 진단용 보존 | 공식 목표 아님, self-collision-avoidance geometry 참고용 |
| Phase 4.5 구 rectangular 실험 | `69276b3` 기반 | (폐기) | 없음 | patch만 보관 | 공식 Sharpa 구현으로 대체된 로컬 WIP |

**중요한 구분**: `69276b3`은 Phase 4.5의 시작점이 아니라 최신
**controller checkpoint**다. Phase 4.5의 실제 시작점은 `d038c5b`
(Sharpa Wave 모델 vendoring/감사가 시작된 커밋)이다. 공식 브랜치 HEAD는
이후 문서·빌드 수정 커밋으로 계속 전진할 수 있으므로 controller
checkpoint와 branch HEAD를 혼동하지 않는다.

## Annotated Tag

| Tag | 대상 커밋 | 의미 |
|---|---|---|
| `phase3-complete` | `849e1b0` | Phase 3 완료 |
| `phase4-start` | `b46fe3b` | Phase 4 시작 (origin/main tip) |
| `phase4-dex3-end` | `bce1dec` | Dex3 연구 종료 — **Gate A 실패**로 종료·대체 |
| `phase4.5-sharpa-start` | `d038c5b` | **Phase 4.5 실제 시작점** — Sharpa Wave vendoring/감사 |
| `phase4.5-sharpa-integration` | `540c9a4` | Stage 2 — G1+Sharpa 통합 완료 |
| `phase4.5-single-hand-prototype` | `a0d286d` | 진단용 단일손 prototype, 공식 목표 아님 |
| `phase4.5-bimanual-prototype` | `69276b3` | 최신 controller checkpoint, **Gate A 미통과** |

## Ancestry 검증 (37차 세션에 실측)

```
git merge-base --is-ancestor bce1dec d038c5b   # YES — bce1dec가 d038c5b의 직접 선행 계열
git merge-base --is-ancestor d038c5b phase4.5/sharpa-wave  # YES
git rev-list --count bce1dec..69276b3   # 당시 Sharpa 전환/controller 커밋 5개
git log --oneline bce1dec..69276b3
#   69276b3 fix(grasp): restore bimanual Sharpa grasp and Gate A semantics
#   a0d286d feat(grasp): add Sharpa multi-finger grasp controller
#   540c9a4 feat(grasp): integrate Sharpa Wave hands with Unitree G1
#   54728b4 docs(grasp): switch Phase 4 target end effector to Sharpa Wave
#   d038c5b feat(grasp): vendor and audit Sharpa Wave hand model
```

공식 브랜치에는 이후 문서·빌드 수정 커밋이 추가되므로
`bce1dec..phase4.5/sharpa-wave`의 개수는 고정된 acceptance criterion이
아니다. 위 5개는 Phase 4.5 전환부터 최신 controller checkpoint까지의
역사적 범위를 고정한 것이다.

## 현재 작업공간 레이아웃

- **유일한 활성 workspace**: `/home/youngjin/Mujoco_humanoid`
- **활성 브랜치**: `phase4.5/sharpa-wave`
- 과거 보조 worktree `/home/youngjin/Mujoco_humanoid_worktrees/
  phase4_5_sharpa`와 `wip/phase4.5-viewer-rectangular` 브랜치는 로컬 구조를
  단순화하기 위해 제거했다.
- 제거 전 dirty 5개 파일의 binary-safe diff가 아래 patch와 정확히 같은
  SHA256임을 확인했다.
  `/home/youngjin/Mujoco_humanoid_local_backups/
  phase4_5_viewer_rectangular_69276b3.patch`
- Dex3 연구 이력은 `phase4/dex3-grasp` 브랜치와 annotated tag에 남아
  있으므로 별도 checkout 디렉터리가 필요 없다.

## 일반 작업 절차

1. 공식 Sharpa 개발은 `/home/youngjin/Mujoco_humanoid`의 clean
   `phase4.5/sharpa-wave` 브랜치에서 수행한다.
2. 새 실험은 별도 폴더/worktree를 자동으로 만들지 말고, 필요성이 명확할
   때만 사용자에게 먼저 이유와 수명주기를 설명한다.
3. Dex3 과거 상태 확인이 필요하면 `phase4/dex3-grasp`의 파일을 `git show`
   등 읽기 전용으로 조회한다. 동시에 checkout해야 하는 명확한 이유가
   없다면 새 worktree를 만들지 않는다.
4. Gate 실패 상태를 성공으로 태그하거나 문서화하지 않는다("Stage/Phase
   완료"는 해당 acceptance criteria를 전부 만족했을 때만 사용).
5. **Phase 5의 whole-body 이동과 IL/BC/PPO는 Phase 4.5의 Gate
   A→B→C→D를 전부 통과하기 전까지 시작하지 않는다.**
   (2026-09-10, 사용자 지시) Phase 5의 **scripted factory 환경**은 이 gate
   앞에서 먼저 구축했다. 이 규칙이 막는 대상인 보행/IL/BC/PPO는 시작하지
   않았고, Phase 4.5의 grasp Gate는 그대로 미통과 상태로 남아 있다.
6. 공식 브랜치에 병합하기 전 반드시: 회귀 테스트 실행 → 결과 보고 →
   실패 시 수정 → 성공 확인 후에만 다음 단계 진행 (project skill의
   Verification rule과 동일).

## Clean-checkout 빌드 전제조건

37차 감사에서 `build_grasp_model_sharpa()`가 참조하는
`GraspEnvConfig.effective_object_half_extents`가 committed config에 없어
clean checkout이 `AttributeError`로 실패하는 문제를 발견했다. 이후
공식 브랜치에 canonical cube만 반환하는 최소 compatibility property와
회귀 테스트를 추가했다. rectangular-object 설정은 WIP 브랜치에만
남기므로 이 수정은 실험 기능이나 Gate 기준을 공식 브랜치로 가져오지
않는다.
