# Whole-Body Humanoid Factory Supervisor

MuJoCo에서 Unitree G1이 공장 자동화의 예외 상황을 복구하도록 만드는
프로젝트다. 정상 생산은 scripted robot arm/conveyor가 담당하고, G1은
dropped part, misalignment, jam 같은 예외에 whole-body로 개입한다.

현재 개발 대상은 G1 + Sharpa Wave 양손의 fixed-base grasp다. Phase 4.5는
아직 미완료이며, SIZE_12에서 Forward Reach Gate를 0.165mm 초과해
Precontact에 도달하지 못했다. 따라서 Gate A(안정 파지) 이후의 hold/lift는
시작하지 않았다. 과거 Dex3 연구는
`phase4/dex3-grasp` 브랜치에 보존한다.

## 현재 실행

Sharpa 자산은 저장소에 직접 포함되지 않는다. 처음 clone한 뒤 설치한다.

```bash
python3 scripts/install_sharpa_wave_assets.py
python3 scripts/test_sharpa_wave_model.py
python3 scripts/test_sharpa_g1_integration.py
```

공식 양손 grasp와 손 동작 demo:

```bash
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --sharpa-hand-demo --no-restart
python3 scripts/test_sharpa_bimanual_grasp.py
```

마지막 테스트에는 아직 달성하지 못한 실제 Gate A 성공 assertion 1개가
의도적으로 실패한다. 이를 threshold 완화로 통과시키지 않는다.

## 문서 안내

- [현재 구조](docs/ARCHITECTURE.md)
- [Phase 0~13 로드맵](docs/PHASE_ROADMAP.md)
- [Sharpa Wave 통합](docs/END_EFFECTOR_SHARPA_WAVE.md)
- [Dex3 legacy 경계](docs/LEGACY_DEX3.md)
- [리팩터링 감사와 파일 분류](docs/REFACTOR_AUDIT.md)
- [Git 브랜치·태그](docs/GIT_WORKFLOW.md)

`PROJECT_CONTEXT.md`는 로컬 작업 인수인계용이며 Git에 올리지 않는다.
세션 원문은 로컬 `docs/history/`에 보존하고 평소 작업에서는 읽지 않는다.

## 고정 개발 순서

각 recovery skill은 다음 순서로 진행한다.

```text
Environment -> Expert -> Demonstrations -> BC -> Evaluation -> PPO
```

최종 목표는 planar-base 데모가 아니라 실제 G1 whole-body locomotion과
manipulation이 결합된 recovery mission이다.
