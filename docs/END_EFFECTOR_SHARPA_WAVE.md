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
- Side-Grasp Posture Gate: **PASS** — 양손이 물체 좌우 측면 바깥에서
  서로 마주보고, 손가락이 아래를 향하는 실제 bilateral side-grasp 자세를
  달성했다(palm inward angle ≈14°, finger-down angle ≈15°, 좌우 mirror
  오차 <0.1mm, forbidden collision 0). `WRIST_SIDE_GRASP_ALIGN` +
  `FOREARM_SIDE_DESCEND`가 과거 `FOREARM_DESCEND`/`WRIST_ALIGN`을 대체한다.
- Precontact/Contact Acquisition: 미도달 — `FOREARM_SIDE_DESCEND`에서
  hand-table collision(최대 약 18.21N, 8N 한계 초과)으로 막힘
- Gate A: FAIL
- Gate B/C/D: 미시도

Gate A는 양손의 thumb + (index 또는 middle) + wrap 대향 접촉이 같은
tick에서 30회 연속 유지되고, object XY 이동 ≤0.03m, peak angular velocity
≤2.0rad/s, forbidden penetration/hand-hand collision 없음까지 만족해야 한다.
기준을 낮춰 통과시키지 않는다.

## Commands

```bash
python3 scripts/install_sharpa_wave_assets.py
python3 scripts/test_sharpa_wave_model.py
python3 scripts/test_sharpa_g1_integration.py
python3 scripts/test_sharpa_hand_demo.py
python3 scripts/test_sharpa_bimanual_grasp.py

DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --sharpa-hand-demo --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --stand
```

`test_sharpa_bimanual_grasp.py`의 실제 Gate A assertion은 성공 전까지
의도적으로 실패한다.
