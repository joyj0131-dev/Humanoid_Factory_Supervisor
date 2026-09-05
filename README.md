# Whole-Body Humanoid Factory Supervisor

MuJoCo에서 Unitree G1이 공장 자동화의 예외 상황을 복구하도록 만드는
프로젝트다. 정상 생산은 scripted robot arm/conveyor가 담당하고, G1은
dropped part, misalignment, jam 같은 예외에 whole-body로 개입한다.

현재 개발 대상은 G1 + Sharpa Wave 양손의 fixed-base grasp다. 기본 실행에서
12cm/0.1kg 블록을 양손으로 잡고 들어올려 공중 5초 유지한다. 실제 물체와
테이블의 간격은 약 8.3cm이며, 추가 5초 유지와 손을 열었을 때 낙하도 검증한다.
이는 양손의 포괄 파지이며, 각 손의 엄지까지 요구하는 기존 Gate A 통과를
뜻하지 않는다. Phase 4.5 전체는 아직 미완료다. 과거 Dex3 연구는
`phase4/dex3-grasp` 브랜치에 보존한다.

현재 접촉에는 손가락뿐 아니라 손바닥·손목도 참여한다. 손끝만의 정밀 파지는
아니다. 최초 버전(`hold_squeeze_m=0.012`)에서는 손바닥+손목이 지지력의 약
84~86%, 손가락은 14~16%뿐이었다 — 팔의 양손 squeeze량이 필요 이상으로
컸기 때문. 실측 스윕 결과 squeeze는 0.002m까지 낮춰도 성공하고(0.001m는
실패) 새 손가락 로직 없이 `hold_squeeze_m=0.003`으로만 낮춰도 손가락
비중이 seed0/1/2에서 약 30~38%까지 오른다(안전마진 3배). soft-contact
관통은 hold 구간 최대 약 4.05mm로 기록한다.

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
OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_grasp_lift.py --seeds 0 1 2
```

`test_sharpa_grasp_lift.py`는 실제 상승·연속 공중 유지·양손 지지·놓았을 때
낙하를 검사한다. 기존 bimanual 테스트에는 미달성 엄지 Gate A 및 과거
실패 상태를 고정한 낡은 assertion들이 남아 있으므로 별도로 보고한다.

접촉 후에는 `SharpaContactLift`가 달성한 손가락 자세를 유지하며 양손을
각각 3mm 더 모으고, 실제 qpos에서 IK를 다시 풀어 1cm/s로 올린다.
MuJoCo `noslip_iterations=10`을 접촉 후 적용해 soft-contact creep를 줄인다.
질량·마찰·충돌 geometry는 바꾸지 않으며 물체 고정/weld/teleport는 없다.
reset 시 solver 설정도 원래 값으로 복구된다.

## 위치·크기·회전 변화 평가

같은 제어기/설정으로 14개 평가 조건과 별도 조합 4개에서 실제 파지·상승·
공중 5초 유지에 성공했다. 수정 전에는 같은 14조건 중 2조건만 성공했다.
시험 조건은 X/Y ±5·10mm, 11/12/13cm 정육면체, 10×15×10cm 직육면체,
yaw ±5°다. 모든 조합을 시험한 것은 아니며 넓은 작업영역 보장이 아니다.
13cm 블록은 최대 관통이 약 9.9mm여서 접촉 품질 개선 대상으로 남는다.
성공한 18조건을 전부 깨끗한 학습 데모로 자동 채택한다는 뜻은 아니다.
자세한 조건·한계·재현법은 [작업영역 평가](docs/SHARPA_WORKSPACE_EVALUATION.md)에 기록한다.

```bash
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --object-size 0.10 0.15 0.10 --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --object-pos-x 0.27 --object-pos-y 0.01 --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --object-yaw-deg 5 --no-restart
OPENBLAS_NUM_THREADS=1 python3 scripts/evaluate_sharpa_workspace.py --suite all --workers 3 --output results/sharpa_workspace/evaluation.jsonl
```

현재는 카메라가 아니라 시뮬레이터의 실제 물체 상태를 사용한다. Sharpa용
모방학습은 아직 시작하지 않았다. 다음은 실행 명령 기록·재생 검증이며,
기존 reach용 BC 파이프라인에 25차원 action만 저장해서 바로 학습하면
preshape/엄지 제어 등 action 밖의 명령을 놓치므로 그대로 재현되지 않는다.

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
