# End-Effector: Sharpa Wave (Phase 4.5 Grasp Track)

> 상태: **미완료.** Stage 0(vendoring/감사)/Stage 1(단독 검증)/Stage 2
> (G1+Sharpa 통합)은 완료·검증됐다. Stage 4(grasp controller)는 파일이
> 구현되어 있고 실제로 실행되지만, **Gate A는 아직 통과하지 못했다** —
> 성공으로 표현하지 않는다. Dex3 비교(Stage 5/6)는 Gate A 통과 전까지
> 시작하지 않는다.
>
> **Phase 4.5의 시작 커밋은 `d038c5b`다.** `69276b3`은 시작점이 아니라
> 이 트랙의 최신 controller checkpoint다. 공식 브랜치는
> `phase4.5/sharpa-wave`이며 이후 문서·빌드 수정으로 HEAD가 전진할 수
> 있다. 역사적 시작점은 계속 `d038c5b`다. 상세 경계/브랜치/태그 표는
> [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) 참고. 세션별 상세 기록은
> `docs/history/PHASE4_GRASP_SESSION_35.md`/`_36.md`/
> `_37_GIT_CLEANUP.md`/`_38_WORKTREE_HYGIENE.md`/`_39.md`/`_40.md`
> (전부 로컬 전용, `.gitignore`).

## 선택 이유

기존 Unitree Dex3-1(3-finger)에서 반복적으로(1~34차 세션, Phase 4)
관측된 엄지 충돌, edge/corner 접촉, force spike, 물체 회전/이탈
문제를 해결하기 위해, 사용자가 2026-08-31(35차 세션)에 end-effector를
5-finger 22-DoF Sharpa Wave로 전환하기로 명시적으로 승인했다. 이
전환이 시작된 커밋이 `d038c5b`이며, 이 시점부터를 **Phase 4.5**로
구분한다(Phase 4는 `bce1dec`에서 Gate A 미통과 상태로 종료·대체됨,
`phase4/dex3-grasp` 브랜치로 별도 보존). G1 본체(어깨/팔꿈치/손목/
허리/다리)는 변경하지 않는다. Dex3-1은 삭제하지 않고 legacy comparison
baseline으로 `phase4/dex3-grasp` 브랜치에 보존한다.

## Upstream / License

- Repository: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
- Pinned commit: `6eea427eb24189519f32b9f21674cd534d3f973c`
- License: Apache License 2.0 (verbatim 보존: `assets/robots/sharpa_wave/LICENSE.txt`,
  `NOTICE.txt`)
- 이 프로젝트에 vendoring된 방식: `assets/robots/sharpa_wave/README.md`
  참고. ~23MB의 MJCF/mesh 서브셋(`left_sharpa_wave/`, `right_sharpa_wave/`)은
  git 이력에 직접 커밋하지 않고, `scripts/install_sharpa_wave_assets.py`가
  pinned commit + checksum 검증으로 재현한다(35차 세션에 클린 상태에서
  재현성 실측 확인: 재다운로드 결과가 원본과 byte-identical). **클린
  worktree를 새로 만든 경우 이 스크립트를 먼저 실행하거나, 이미
  다운로드된 원본 workspace의 해당 디렉터리를 심볼릭 링크해야 모델이
  컴파일된다** — 37차 세션의 `phase4_5_sharpa` clean worktree는 원본
  workspace(`/home/youngjin/Mujoco_humanoid`)로부터 심볼릭 링크했다.

## Model structure (컴파일된 MjModel에서 실측, 35차 세션)

- 손 1개당 22 active joint / 22 `<position>` actuator: thumb(CMC_FE,
  CMC_AA, MCP_FE, MCP_AA, IP=5), index/middle/ring(MCP_FE, MCP_AA, PIP,
  DIP=4씩), pinky(CMC, MCP_FE, MCP_AA, PIP, DIP=5) — 5+4+4+4+5=22,
  벤더 스펙과 정확히 일치.
- 좌/우 완전 대칭(joint 이름이 `left_`/`right_` 접두사만 다름, diff로 확인).
- `_with_wrist` variant는 **관절이 아닌 강체(rigid) wrist 스탠드오프
  geom**(`wrist_B.STL`/`wrist_collision.STL`)을 루트 body에 추가해
  손 자체 geometry를 로컬 Z로 +29mm 이동시킬 뿐, 추가 자유도는 없다.
  `_with_flange` variant는 이 스탠드오프가 없다.
- 손 1개 총 질량: 1.2477kg (23개 body 합산, `world` 제외).
- Neutral pose(`qpos0=0`)는 모든 joint range 안에 있음(일부는 경계값
  — 예: `pinky_CMC`, `*_PIP` range 하한이 0이라 정확히 그 값).

## G1 mounting transform — Stage 2 완료 (540c9a4)

G1+Sharpa 통합 모델이 실제로 컴파일된다: 자기충돌 0, Sharpa 손 무게
하중에서 arm 처짐 없음, 통합 테스트 스위트(`test_sharpa_g1_integration.py`)
통과. 좌/우 각 손은 `attach_sharpa_hands()`(`model_builder.py`)가
`prefix=f"{side}_"`로 부착하며, 벤더 XML 자체가 이미 `left_`/`right_`
접두사를 갖고 있어 최종 body/joint/actuator 이름이 `left_left_...`/
`right_right_...`로 이중 접두사가 된다(오타 아님, 의도된 결과 — 상세는
`humanoid_learning/envs/sharpa_config.py` 모듈 docstring 참고).

> **[37차 감사 후 수정한 빌드 전제조건]** `model_builder.
> build_grasp_model_sharpa()`가 540c9a4부터 참조해 온
> `GraspEnvConfig.effective_object_half_extents`가 committed config에 없어
> clean checkout이 `AttributeError`로 실패하는 문제를 확인했다. 공식
> 브랜치에는 canonical cube의 세 축을 반환하는 최소 property와 회귀
> 테스트만 추가했다. WIP의 `object_half_extents` 직육면체 옵션은 가져오지
> 않았으므로 controller/IK/Gate 기준과 canonical geometry는 불변이다.

## Action mapping / 전체 G1+Sharpa 차원 — 실측 완료

`SharpaGraspEnv`(25-dim action: waist 3 + arm 14 + 좌우 각 4-group
synergy 8) / observation 129-dim(양팔 qpos·qvel + 양손 4-group curl
qpos·qvel + 양 palm pose(pos+9-dim rotation) + object pos·quat·linvel·
angvel + per-(side,group) net contact force) — 36차 세션에 실제 컴파일된
모델에서 실측, 테스트로 shape 고정(`test_sharpa_bimanual_grasp.py`).

## Tactile/contact 모델의 실제 한계 (실측, 35차 세션)

Sharpa Wave의 MJCF(`left_sharpa_wave.xml`, `_with_wrist`, `_with_flange`,
좌/우 전부)에는 **`<sensor>` 엘리먼트가 하나도 없다** — `grep`으로 직접
확인. 실물 하드웨어의 240×240 dynamic tactile array나 slip inference는
MuJoCo 모델에 존재하지 않는다. 존재하는 것은 각 fingertip의
**elastomer collision geom**(`*_elastomer`, contype=1, 소프트한
`solref` — thumb는 `[0.06, 0.9]`, 나머지 4손가락은 `[0.02, 1.0]`)뿐이다.
이 프로젝트가 사용할 모든 손가락별 힘/접촉 관측은 MuJoCo의
`mj_contactForce`/`data.contact`에서 직접 유도해야 하며, **"contact-force
기반 관측"이라고 명시**해야 한다 — 하드웨어 tactile array를 모사했다고
서술하지 않는다.

## Grasp controller — Stage 4 구현됨, Gate A 미통과

두 개의 서로 다른 controller가 존재하며 절대 혼동하지 않는다.

- **`SharpaSingleHandGraspExpert`**(`humanoid_learning/expert/sharpa_grasp_expert.py`,
  35차 세션 원본, 36차에 진단용으로 재라벨링) — 오른손만 사용, 왼손은
  stand pose 유지. **공식 목표가 아니다.** thumb/middle/wrap 그룹이
  실제 contact-force 기반 접촉(1.7~4.9N)까지는 달성했으나 FORCE_SETTLE
  중 CONTACT_LOST로 실패. Dex3와 직접 비교 금지.
- **`SharpaBimanualGraspExpert`**(`humanoid_learning/expert/sharpa_bimanual_grasp_expert.py`,
  36차 세션 신규, 커밋 `69276b3`) — **공식 Phase 4.5 목표.** 양팔을
  `CoupledBilateralIK`로 동시에 푼다. WRIST_ALIGN까지는 실측
  orientation drift 0.045°(허용 5°)로 안정적으로 수렴한다. 그러나
  FINGERTIP_PRECONTACT 이후 IK가 보고하는 위치 오차(<1cm)와 실제
  물리로 추종시켰을 때 손바닥이 정착하는 Cartesian 위치 사이에
  재현 가능한 5~6cm 간극이 있어, 완전히 오므린 손가락도 물체에 닿지
  못한다 — `CONTACT_ACQUIRE`가 양쪽 모두 접촉 0회로 TIMEOUT.
  `max_bilateral_stable_streak = 0/30`. **Gate A: 미통과.**

  > **[39차 세션 — 원인 규명 및 부분 완화, Gate A 여전히 미통과]**
  > 이 5~6cm 간극을 layer-by-layer로 계측한 결과(`scripts/
  > diagnose_precontact_gap.py`), ctrl register(`env._arm_target`)는
  > IK가 푼 joint target에 정확히 수렴하는데(`joint_target_minus_ctrl_norm
  > == 0`) 실제 물리 qpos만 뒤처지는 **steady-state compliant-actuator
  > (arm_kp=120) 중력 부하 tracking error**로 확정했다(홀드 시간을
  > 3000+ tick으로 늘려도 간극이 줄지 않는 진짜 asymptotic floor임을
  > 확인). Dex3의 검증된 `_coupled_maybe_resolve`(Cartesian-target
  > inflation resolve) 방식을 그대로 이식해봤으나 이 reach에서는
  > causal A/B로 **오히려 악화됨을 확인**(6.9cm → 10~25cm, joint
  > norm > 1rad) — 사용하지 않는다. 대신 `data.qfrc_bias/arm_kp`를
  > 매 tick ctrl에 더하는 물리 기반 feedforward(`GraspEnvConfig.
  > arm_gravity_compensation`, 기본값 False, `SharpaBimanualGraspExpert`
  > 전용 env에서만 활성화)로 간극을 약 6.9cm → 3.6~4.3cm로 절반 가까이
  > 줄였다(causal A/B 테스트로 검증, `test_arm_gravity_compensation_
  > reduces_precontact_tracking_error`). Joint frictionloss(0.3, 전
  > 팔 관절 확인됨) 보정도 시도했으나 기여가 무시할 수준(수 mm)임을
  > 확인해 반증·폐기했다. FINGERTIP_PRECONTACT는 이제 고정 tick 수가
  > 아니라 실측 palm pose 기반 Precontact Tracking Gate(위치 ≤1cm,
  > orientation ≤5°, 15-tick 연속)로만 다음 상태로 전이하며, 실패 시
  > `BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED`로 정직하게
  > 보고한다. **이 Gate는 아직 통과하지 못했다** — 남은 ~3.6~4.3cm는
  > 아직 원인 미확정이며(qfrc_constraint 계측에서 어깨 관절에 최대
  > ~17.5Nm의 큰 constraint force가 관측됨 — 같은 손 내부 self-collision
  > 후보로 추정되나 확정되지 않음), 다음 세션의 단일 blocker다. 상세:
  > `docs/history/PHASE4_GRASP_SESSION_39.md`.

Gate A 정의(Dex3의 `_bilateral_tripod_streak` 패턴을 그대로 모델링):
양측 동시에 thumb 접촉 AND (index 또는 middle) 접촉 AND wrap 접촉
AND thumb의 힘 방향이 index/middle 합력과 실제로 반대(dot<0) AND
hand-hand 비접촉이 30-tick 연속 유지 AND object XY 변위 ≤0.03m(Dex3
`object_displacement_limit`과 동일값, 코드로 확인) AND object 각속도
peak ≤2.0rad/s(신규 공개값 — Dex3에 코드화된 각속도 기준 없음, `grep`으로
확인) AND 금지된 침투/hand-hand 힘 없음 AND 종료 상태가 FAILURE 아님.
`ever_contacted`만으로는 절대 통과하지 않는다(테스트로 검증,
`test_sharpa_bimanual_grasp.py::test_ever_contacted_alone_does_not_satisfy_gate_a`).

## Dex3 비교 방법

**미완료.** 첫 SIZE_12(12cm 정육면체, half=0.06m) canonical grasp 비교는
Sharpa fixed-base Gate A를 통과한 뒤에만 실행한다(사용자 지시 Stage 5/6).
비교 표 형식은 사용자 지시의 Stage 6 지표 목록을 그대로 따른다.

## Viewer / 테스트 명령

```
# Sharpa Wave 손 모델 감사 (읽기 전용, 컴파일된 MjModel 정보 출력)
python3 scripts/audit_sharpa_wave.py

# Sharpa Wave 단독 모델 검증 (9개 테스트)
python3 scripts/test_sharpa_wave_model.py

# G1+Sharpa 통합 검증 (6개 테스트)
python3 scripts/test_sharpa_g1_integration.py

# Sharpa Wave 자산 설치/재현 (클론 직후 1회)
python3 scripts/install_sharpa_wave_assets.py

# 단일손 진단 / 공식 양손 controller 테스트
python3 scripts/test_sharpa_single_hand_diagnostic.py
python3 scripts/test_sharpa_bimanual_grasp.py

# FINGERTIP_PRECONTACT IK-vs-physics gap 계층 분해 진단 (39차 세션)
python3 scripts/diagnose_precontact_gap.py

# Sharpa mount(_with_wrist vs _with_flange) 감사·A/B (40차 세션)
python3 scripts/audit_sharpa_mount.py

# 실제 MuJoCo viewer (39차부터 공식 브랜치에 존재, 37차의 "미커밋" 기록은 낡음)
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --hand-model sharpa --no-restart
```

> **[37차 세션 — Git 정리, 낡은 기록]** 이 문단이 최초 작성된 시점에는
> `--hand-model sharpa`/`mode_grasp_sharpa()`가 WIP 브랜치에만
> 있었다. **39차 세션(커밋 `5afdab2`)에 공식 브랜치로 독립
> 재구현·커밋됐고, 40차 세션이 `--sharpa-mount`/`--sharpa-visual-style`을
> 추가했다** — 더 이상 미커밋 상태가 아니다. WIP의 10×15×10cm
> 직육면체 실험은 여전히 WIP 브랜치 전용으로 남아 있다(가져오지 않음).

> **[40차 세션 — mount 감사, object-facing orientation(opt-in),
> self-collision 안전장치, 외형 통일]** `_with_wrist` vs `_with_flange`
> mount는 실측상 사실상 동일함을 확정했다(palm 위치 차이 0.000mm,
> fingertip reach 차이 0.48mm — 기본값 `sharpa_mount="wrist"` 유지).
> "부자연스러운 자세"의 실제 원인은 mount가 아니라 WRIST_ALIGN이
> orientation "안정성"만 확인하고 "정확성"(물체를 향하는지)은 확인하지
> 않았던 것이었다(index/middle fingertip이 물체에서 12~21cm 벗어남을
> 실측). 명시적 object-facing target(`_object_facing_R`)을 구현했으나,
> **causal A/B로 이 target이 `torso_link<->right_wrist_pitch_link`
> self-collision(최대 113N)을 유발함을 확정**했다 — 39차가 확정한
> 것과 같은 범주(steady-state actuator physical tracking error)가
> orientation에도 적용됨을 보여준다. 이 문제를 이번 세션에서 완전히
> 해결하지 못해, `BimanualGraspConfig.object_facing_orientation`
> 플래그(기본값 **False**)로 게이팅해 공식 기본 동작(39차 검증 결과)은
> 전혀 변경하지 않았다 — opt-in으로만 존재한다. 새로운
> `SharpaGraspEnv._torso_arm_collision_force()` 안전장치는 플래그와
> 무관하게 항상 계산되며, 39차 기본 경로에서는 0N임을 확인했다(무회귀).
> Sharpa 외형은 G1 자체 material(metal/black)로 통일했다(물리 불변
> 검증됨, 기본값 `sharpa_visual_style="upstream"`이나 viewer는
> `g1`을 기본 사용). 상세: `docs/history/PHASE4_GRASP_SESSION_40.md`.
