# G1 + Sharpa Wave

Sharpa Wave는 활성 브랜치의 유일한 end-effector다. 과거 손 연구는
`phase4/dex3-grasp`에 보존되며 active runtime에는 hand-model selector가
없다.

## Asset and mount

- G1 base: local menagerie `assets/robots/g1/g1.xml`
- Sharpa metadata/installer: `assets/robots/sharpa_wave/`,
  `scripts/install_sharpa_wave_assets.py`
- attachment: `model_builder.attach_sharpa_hands()`
- default mount: `wrist`; `flange`는 0.5mm 차이의 진단 대안
- default visual style: G1의 실제 `black`/`metal` material

Builder는 bare G1에서 시작한다. pre-hand-equipped model을 사용하며 남던
손목당 0.202839kg의 ghost hand mass는 제거됐다. attachment 후 stand
keyframe은 joint/actuator name으로 복구한다.

검증된 compiled contract:

- `nq=80`, `nv=79`, `nu=73`
- Sharpa actuator 44개(손당 22)
- stand self-collision 0
- 좌우 fingertip mirror error 0.1mm 미만
- compliant arm에서 3초 wrist drift 7.394mm

## Control contracts

Sharpa 손은 각 손마다 네 closing group을 사용한다.

```text
thumb | index | middle | wrap(ring+pinky)
```

- WholeBodyEnv: legs12 + waist3 + arms14 + hands8 = action 37
- SharpaGraspEnv: waist3 + arms14 + hands8 = action 25, observation 129
- preshape/abduction joints는 phase target으로 별도 유지
- contact force safety는 각 physics substep에서 검사

공식 controller는 `SharpaBimanualGraspExpert` 하나다. free-space 손 검증은
`SharpaHandDemo`가 담당하며 Gate 결과로 세지 않는다.

## Gate status

Phase 4.5는 미완료다.

작업영역 확장(2026-09-05): 동일한 제어기로 위치·크기·yaw 14조건과 별도
조합 4조건의 실제 상승/5초 유지 검증 완료. 손목을 누락하던 지지력 집계를
수정했으며, 손가락만의 파지로 재분류하지 않는다. 모델 질량·마찰·장착·
관절은 변경하지 않았다. [평가 조건/재현법](SHARPA_WORKSPACE_EVALUATION.md) 참고.

- Natural Posture Gate: PASS
- Wrist Transition Gate: FAIL (`5.125rad/s > 2.0rad/s`)
- Forward Reach Gate: PASS (settled palm error 약 9.96mm, streak 15,
  collision 0 — waypoint 스케줄 버그 수정으로 해결)
- 접근: horizontal-wrap 정렬 후 CONTACT_ACQUIRE에서 실제 양손 접촉
- 엄지-specific Gate A: FAIL (정의 유지, 실제 lift 시도의 선행조건에서는 분리)
- 기본 양손 포괄 파지: 2초 hold, 실제 상승, table clearance ≥5cm에서 공중 5초 유지 성공
- 기본 12cm/0.1kg 장면에서 clearance 약 8.3cm, 추가 5초 유지 및 놓기 검증

Gate A는 양손의 thumb + (index 또는 middle) + wrap 대향 접촉이 같은
tick에서 30회 연속 유지되고, object XY 이동 ≤0.03m, peak angular velocity
≤2.0rad/s, forbidden penetration/hand-hand collision 없음까지 만족해야 한다.
기준을 낮춰 통과시키지 않는다.

`physical_grasp_success`는 이 엄지-specific Gate와 별개의 실제 물체 지지/상승
결과다. 손가락 자세를 무조건 더 닫지 않고 유지하면서 양팔을 함께 올린다.
접촉 후 MuJoCo `noslip_iterations=10`을 적용한다(마찰계수/질량/geometry
불변). 손을 열면 물체가 낙하하는 테스트로 고정/부착 없는 물리 접촉임을 검증한다.

**지지력 구성**: 접촉 지점을 물체 로컬 좌표로 보면 왼손은 +Y면, 오른손은
-Y면을 누르고 있어 양손이 서로 반대쪽에서 물체를 사이에 끼우는 협동
(vise-like) 구조다. 엄지는 접촉력이 거의 0 — 한 손 안에서 엄지가
index/middle에 대립하는 tripod 구조(Gate A가 요구하는 것)는 아니다. 사용자
확인: 이 프로젝트는 애초에 양손 협동 grasp이 기본 설계이므로 이 자체는
문제가 아니다.

**손가락 지지 비중 개선**: 원래 `hold_squeeze_m=0.012`에서는
손바닥(hand_C_MC)+손목(wrist_yaw_link)이 지지력의 84~86%, 손가락은
14~16%뿐이었다. 손가락에 추가로 닫는 힘을 얹는 시도 2건(THUMB_OPPOSE
시간 연장, 이미 닿은 그룹만 살짝 더 조이기)은 둘 다 이미 성공하던 rollout을
CONTACT_LOST로 회귀시켰다 — 12mm squeeze와 얼린 손가락 자세가 아주
미세하게 맞춰진 균형점이었다. 대신 `hold_squeeze_m` 자체를 스윕한 결과
0.002m까지는 성공하고(0.001m는 실패) 원래 0.012m가 필요 이상으로 컸다는 게
드러났다. `hold_squeeze_m=0.003`(실패 지점 대비 3배 여유)으로 낮추는 것만
으로 새 손가락 로직 없이 손가락 비중이 seed0/1/2 모두 약 30~38%까지 오른다.

## Commands

```bash
python3 scripts/install_sharpa_wave_assets.py
python3 scripts/test_sharpa_wave_model.py
python3 scripts/test_sharpa_g1_integration.py
python3 scripts/test_sharpa_hand_demo.py
python3 scripts/test_sharpa_bimanual_grasp.py
OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_grasp_lift.py --seeds 0 1 2

DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --sharpa-hand-demo --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --stand
```

기존 bimanual 테스트의 엄지 Gate A 및 과거 실패 결과 assertion과, 새 실제
lift 회귀 결과는 구분하여 보고한다.
