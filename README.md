# Whole-Body Humanoid Factory Supervisor

MuJoCo에서 Unitree G1(양손 Sharpa Wave)이 공장 자동화의 예외 상황을 복구하도록
만드는 프로젝트다. 정상 생산은 scripted 자동화 팔과 컨베이어가 담당하고, G1은
떨어진 부품·정렬 불량·걸림 같은 예외에 전신(whole-body)으로 개입한다.

```text
고장 감지 → 라인 정지·복구 요청 → G1 이동 → 전신 자세 → 물리 복구 → 복구 검증 → 라인 재가동
```

## 현재 상태 요약 (2026-09-28, seed 0)

| 기능 | 상태 | 실측 |
|---|---|---|
| Line 1 벨트 걸림: 보행 → 양손 파지 → 상승 → 5초 유지 | 성공 | 4934 step, 79.5 mm 상승 |
| Line 2 바닥 낙하(기본 파지): 보행 → 앉기 → 바닥 집기 → 기립 → 5초 유지 | 성공 | 8760 step, 601.7 mm 상승 |
| Line 2 바닥 낙하(검은 패드 + 무릎 벌림, `--pad-grip --frog-stance`) | 성공(실험 옵션) | 9576 step, 438 mm, 유지 중 하중 전부 패드 |
| Line 2 바닥 → 테이블 안착 → 손 회수 → 복구 검증 → 라인 재가동 | **미완료** | 판단 루프는 동작, 물리 경로 미확보(아래 참고) |
| 운반 보행(블록을 든 채 걷기) | 미확보 | 팔을 앞으로 든 자세에서 기존 보행 정책이 명령을 추종하지 못함 |

"성공"은 해당 장면에서의 기능 성공이다. 충돌 없는 동작, 범용 복구 정책,
학습 데모 승인을 뜻하지 않는다. 물체 고정·순간이동·숨은 외력·충돌 비활성화·
성공 기준 완화는 쓰지 않는다. 로봇은 액추에이터 명령으로만 움직인다.

## 설치

Sharpa 손 자산과 G1 보행 정책(Unitree 사전학습, BSD-3, 외부 도구)은 저장소에
직접 포함되지 않는다. 처음 clone한 뒤 설치한다.

```bash
python3 scripts/install_sharpa_wave_assets.py
python3 scripts/install_g1_walk_policy.py
python3 scripts/test_sharpa_wave_model.py
python3 scripts/test_sharpa_g1_integration.py
```

보행은 연구 대상이 아니라 주어진 도구이며, 출처는 `assets/policies/g1_walk/NOTICE`에 있다.

## 두 작업 공간 공장 환경

1.3 m 통로 양쪽에 독립된 두 라인이 있고, 각 라인은 컨베이어·자동화 팔·테이블을
하나씩 쓴다. Line 1(index 0)은 벨트 `jam`, Line 2(index 1)는 자동화 팔이 부품을
테이블 밖 바닥으로 떨어뜨리는 `arm_drop` 고장이다. Task manager가 라인을 멈추고
G1에 복구를 요청한다. 복구 완료는 신호만으로 인정하지 않는다. 부품이 지정 위치
(허용 60 mm, yaw ±0.15 rad는 90° 대칭 기준)에 기울지 않고 정지 상태로 작업면에
놓인 것을 물리적으로 50 tick 확인한 뒤에야 팔과 벨트를 다시 가동한다.

```bash
DISPLAY=:0 python3 scripts/view_factory.py                                   # 고장 장면
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 0      # Line 1 복구
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1      # Line 2 복구(기본 파지)
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1 --pad-grip --frog-stance
OPENBLAS_NUM_THREADS=1 python3 scripts/view_factory.py --offscreen \
  --seed 0 --out results/factory/scene.png --steps 400 --capture 150 400
```

Navigation Gate는 보행 정책 도입 전에 정해 두었다. 위치 0.10 m, heading 0.15 rad,
1.0초 유지, 넘어짐·금지 접촉 없음, 올바른 셀 먼저 방문, 1500 step 이내 조건이다.
step당 base 이동이 0.05 m를 넘으면 순간이동으로 보고 실격시킨다.

### Line 1 (벨트 걸림)

실제 보행 → 양손 파지 → 79.5 mm 상승 → 공중 5초 유지(4934 step, seed 0/1).
손–블록 최대 관통은 7.6 mm이며 확장 간섭 검사도 0 tick은 아니다.

### Line 2 (바닥 낙하) — 기본 파지

실제 고장 → 보행 → 앉기 → 엄지를 제외한 바닥 집기 → 기립 → 5초 유지
(8760 step, 최대 바닥 간격 601.7 mm, 발–블록 접촉 0, 넘어짐 없음). 앉을 때는
엄지를 접어 두고, 손을 내리는 후반에 편다. 손목을 높게 두고 손가락을 위에서
양옆으로 내리는 바닥 전용 접근이다. 네 손가락이 하중을 균등하게 받치지는 않는다.

### Line 2 — 검은 패드 + 무릎 벌림 (`--pad-grip --frog-stance`, 실험 옵션)

손가락 끝 두 관절을 거의 편 채 elastomer 패드 면으로 블록 옆면을 잡는다. 앉을 때는
발끝을 돌리고 무릎을 벌린다(실측 무릎 폭 약 22 → 39 cm). 기립 중에는 손을
실제 어깨 좌표계에서 추종하고, 명령 손 위치를 기준으로 서보해 squeeze를 유지한다.

- 실측(seed 0): 9576 step에 LIFTED, 바닥 간격 438 mm, 약 11.9초 직립 유지,
  접촉 유실 0 tick, 손–물체 관통 3.1 mm. 유지 중 손–물체 하중은 전부 패드였고
  엄지·손가락 껍질 하중은 0이었다.
- 초기 조건 변형(부품 x ±2 cm, y −2 cm, 로봇 ±5 cm·±0.1 rad) 5개도 LIFTED.
- 남은 문제: 부품 +2 cm y(블록 yaw 32.6°)는 PICK_CLEAR에서 실패한다.
  어깨–몸통 접촉(최대 9.5~33 N)도 남아 있다.
- 패드 힘 측정은 단단한 손가락 껍질의 접촉력을 제외한다. 이 옵션은 손가락 명령
  매핑을 바꾸므로 기존 데모와 같은 학습 계약으로 보지 않는다.

### Line 2 — 바닥 → 테이블: 관측·선택·실행·검증·재계획 루프 (`--table-place`, 미완료)

바닥 블록을 테이블 지정 위치에 놓고 라인을 재가동하는 전체 복구를 위해, 기존
복구 FSM을 유지한 채 그 둘레에 폐루프 판단 구조를 붙였다(`humanoid_learning/expert/recovery_agent.py`,
`factory_floor_table.py`). LLM이나 외부 API는 쓰지 않는다. 선택 담당은
인터페이스로 분리해 나중에 교체·비교할 수 있게 했다.

- **관측**: 블록의 실제 위치·방향·속도, pelvis 기울기, 양발 하중, 패드 하중,
  손–다리 거리, 계획 오차.
- **후보 평가**(실제 데이터를 건드리지 않는 복사본에서): 안착 도달 → 파지 자세 도달 →
  들어 올려 테이블 앞면을 넘기는 **이송 경로** → 손 **진입 경로** 순으로 연결을 확인한다.
  끝점만이 아니라 경로 전체를 보고, 손–다리·손–블록 거리와 경로 도중의
  팔 해 뒤집힘(분기 전환)도 검사한다. 경로 허용치는 물리적으로 성공한 기존 진입
  경로를 같은 검사기로 결정 때마다 측정해 보정한다.
- **선택**: 모든 연결이 되는 후보만 고른다. 없으면 이유를 붙여 제어된 중단(ABORT)을 한다.
- **실행 중 감시와 재계획**: 블록이 밀리거나 돌거나, 손이 다리를 계속 밀거나,
  계획 팔 자세가 급변하거나, 패드 하중이 형성되지 않으면 실패로 본다. 이때 이미
  안정하게 지나온 진입 경로를 거꾸로 되감아 물러난 뒤, 다시 관측하고 실패한
  계획을 빼고 재선택한다(최대 3회).
- **기록**: 판단마다 관측값, 후보별 탈락 이유, 선택과 근거, 실행 결과를 남긴다.

물리로 확인된 것:
- 기존 진입 자세를 거친 손 진입, 8개 패드 접촉(껍질 지지 아님), 바닥 위에서 블록을
  −31~−34° 돌려 테이블 축에 정렬(잔차 3~4°), 3 cm 들어 올림.
- 감시가 실제로 동작함: 블록 13 mm 밀림을 감지해 물러난 뒤 재판단했다. 넘어짐 없음.

막힌 원인: 이 장면에서 패드 파지 하나로 바닥 파지와 테이블 앞면 통과를 함께
만족할 수 없다.
- 웅크린 채 바닥 블록을 잡으려면 손가락이 수평에서 약 55° 이상 아래를 향해야 한다.
- 발을 고정한 채 블록을 몸 가까이 높이 들어 테이블 앞면을 넘기려면 약 50° 이하가 필요하다.
- 이 판단이 들어가기 전 실행한 기립 4회는 모두 같은 구간(블록 높이 0.45~0.5 m)에서
  팔 관절 한계로 파지를 잃었다. 지금은 선택기가 이를 실행 전에 판단해 ABORT하고,
  로봇은 선 채로 멈추며 블록은 바닥에 그대로 있다.

운반 보행의 근거 범위: 블록 없이 팔을 앞으로 든 자세만 두고 시험했다. 0.3 m/s 명령
1.5초 동안 거의 전진하지 않다가 명령 없이 0.15 m 밀려 나갔고, 피치 12°에서 정지하지
못했다. 팔을 내리면 정상 보행한다. 실제 블록을 든 보행, 명령 신호 조정, 정책 입력
보정은 시험하지 않았다. 다음 단계에는 발 위치를 바꾸는 수단(준정적 한 걸음 제어기,
운반 자세 보행 재학습, 손 안 재파지 중 하나)이 필요하며 결정 대기 중이다.

```bash
DISPLAY=:0 python3 scripts/view_factory.py --recover --fault-workcell 1 \
  --pad-grip --frog-stance --table-place          # 종료 시 판단 기록 출력
python3 scripts/test_recovery_agent.py
```

`--table-place`는 기본값 OFF다. 옵션을 켜지 않으면 Line 1과 Line 2의 궤적
(qpos/qvel/ctrl)은 이전 커밋과 매 tick 비트 단위로 같다.

### 성능 옵션

- `--display-hz`(기본 50): 창 갱신 주기만 줄인다. 물리·제어는 매 tick 실행한다.
- `--posture-solve-interval`(기본 1): 바닥 자세 IK를 N tick마다 다시 푼다. 명령과
  균형·접촉 피드백은 매 tick 유지된다. N=2는 기본 파지에서 성공을 유지했지만
  패드 경로에서는 접촉을 더 일찍 잃어 기본값으로 채택하지 않았다.
- 접촉 집계는 한 번의 순회로 처리하며, 궤적은 변경 전과 비트 단위로 같다.

## 기반 작업: 고정 몸통 양손 파지 (Phase 4.5)

G1 + Sharpa Wave 양손으로 12 cm / 0.1 kg 블록을 잡고 들어 올려 공중 5초 유지한다
(물체–테이블 간격 약 8.3 cm, 손을 열면 낙하까지 검증). 이는 양손의 포괄 파지이며,
각 손의 엄지까지 요구하는 기존 Gate A 통과를 뜻하지 않는다. 손바닥·손목도 접촉에
참여한다. `hold_squeeze_m=0.003`에서 손가락 하중 비중은 약 30~38%(seed 0/1/2)다.
Dex3 연구는 `phase4/dex3-grasp` 브랜치에 보존한다.

```bash
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --free-base --no-restart   # 두 발로 선 채
DISPLAY=:0 python3 scripts/view_whole_body.py --sharpa-hand-demo --no-restart
OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_grasp_lift.py --seeds 0 1 2
```

- 위치·크기·회전 변화: 같은 제어기로 14개 평가 조건과 조합 4개에서 파지·상승·5초
  유지에 성공했다(수정 전 14개 중 2개). X/Y ±5·10 mm, 11/12/13 cm 정육면체,
  10×15×10 cm 직육면체, yaw ±5° 범위이며 넓은 작업영역 보장은 아니다.
  13 cm 블록은 관통 약 9.9 mm로 개선 대상이다.
  (`scripts/evaluate_sharpa_workspace.py --suite all`)
- 데모 기록·재생: 25차원 action만이 아니라 preshape·엄지 명령과 solver 설정을 포함한
  완전한 명령을 기록한다. Expert 없이 저장된 명령만 실행해 1e-6 이내로 재현된다.
  파일럿 10개는 학습용으로 승인된 데이터가 아니다(`learner_ready=False`).
  (`scripts/sharpa_demos.py collect|replay`)

## 테스트

```bash
python3 scripts/test_factory.py
python3 scripts/test_factory_recovery.py
python3 scripts/test_floor_pad_planning.py
python3 scripts/test_factory_place.py
python3 scripts/test_recovery_agent.py
```

알려진 문제: `scripts/test_factory_task_frames.py`는 테스트의 가짜 객체가 현재 바닥
파지 코드보다 오래되어 실패한다(이번 변경 이전부터 동일).

## 문서와 기록

설계 메모, 실험 기록, 세션 인수인계(`PROJECT_CONTEXT.md`, `docs/`)는 로컬 전용이며
Git에 올리지 않는다. 저장소에 올리는 Markdown은 이 README 하나다.

## 고정 개발 순서

각 recovery skill은 다음 순서로 진행한다.

```text
Environment -> Expert -> Demonstrations -> BC -> Evaluation -> PPO
```

최종 목표는 planar-base 데모가 아니라, 실제 G1 whole-body locomotion과
manipulation이 결합된 recovery mission이다.
