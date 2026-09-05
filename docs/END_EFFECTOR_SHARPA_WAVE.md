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
