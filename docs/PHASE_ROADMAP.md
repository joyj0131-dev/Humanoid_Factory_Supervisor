# Phase Roadmap

이 문서는 Phase 0~13의 방향과 현재 상태를 빠르게 확인하기 위한 공개
로드맵이다. 세션별 실패 실험은 포함하지 않는다.

| Phase | 상태 | 목표 |
|---|---|---|
| 0 | 완료 | Architecture review |
| 1 | 완료 | MuJoCo G1 manipulation foundation |
| 2 | 완료 | Bimanual reach expert foundation |
| 3 | 완료 | 50-demo dataset, BC pipeline, evaluation smoke test |
| 4 | 미완료 | Whole-body G1 foundation과 grasp/lift 검증 |
| 4.5 | 활성·미완료 | Sharpa Wave fixed-base bimanual grasp Gate A→D |
| 5 | 미착수 | Scripted Factory Automation Environment |
| 6 | 미착수 | Recovery #1 Dropped Part expert |
| 7 | 미착수 | Recovery #1 demonstrations + BC |
| 8 | 미착수 | Recovery #1 evaluation + PPO fine-tuning |
| 9 | 미착수 | Recovery #2 Misaligned Part 전체 파이프라인 |
| 10 | 미착수 | Recovery #3 Workcell Jam 전체 파이프라인 |
| 11 | 미착수 | Multi-exception supervisor integration |
| 12 | 미착수 | Full factory mission evaluation |
| 13 | 미착수 | Portfolio, README, result analysis |

## Phase 4.5 acceptance order

1. Wrist Transition / Natural Posture
2. Forward Reach / Orientation Alignment
3. Precontact Tracking / Contact Acquisition
4. Gate A — 안정 파지
5. Gate B — 2초 tabletop hold
6. Gate C — 5cm lift
7. Gate D — 5초 air hold

현재 확인 상태(2026-09-05):

작업영역 확장 검증: 수정 전 2/14 → 수정 후 14/14 조건 성공, 별도 위치·
크기·회전 조합 4/4 성공. 유한한 조건 검증이며 범용 파지 완료가 아니다.
[정확한 조건 및 한계](SHARPA_WORKSPACE_EVALUATION.md)를 따른다.
다음 구현은 Sharpa 명령 기록/재생 검증 후 소규모 state-based BC다.
카메라 인식·BC/PPO는 이번 작업에서 구현하지 않았다.

- Mount Integration Gate: PASS
- Natural Posture Gate: PASS
- Wrist Transition Gate: FAIL (`5.125rad/s > 2.0rad/s`)
- Forward Reach Gate: PASS (`9.96mm ≤ 10mm`, 15-tick streak, 금지 충돌 0건)
- 접근: horizontal-wrap 자세에서 실제 CONTACT_ACQUIRE 진입
- 엄지-specific Gate A: FAIL (`0/30`), 정의 유지
- 실제 양손 포괄 파지: hold 2초 → 상승 → table clearance ≥5cm에서 5초 유지 성공
- 실제 clearance 약 8.3cm, 추가 5초 유지 및 actuator로 손을 열면 낙하 검증
- 전체 Phase 4.5 완료 선언은 아님. 다양한 물체/접근 및 엄지 topology 검증은 남음

사용자 요청에 따라 엄지-specific Gate A를 실제 lift 시도의 선행조건에서
분리했다. 기존 Gate A를 통과했다고 재명명하지 않고 `physical_grasp_success`로
실제 지지·상승 결과를 별도 보고한다. Factory/IL/BC/PPO는 이번 변경 범위 밖이다.

## Final target

최종 목표는 G1이 실제 whole-body locomotion과 manipulation을 사용해
fault 발생 지점으로 이동하고, 자세를 조절하고, 복구하고, 완료를 검증하는
mission 전체다. Planar base는 개발·디버깅 fallback일 뿐 최종 목표를
대체하지 않는다.

Recovery skill마다 순서는 고정한다.

```text
Environment -> Expert -> Demonstrations -> BC -> Evaluation -> PPO
```
