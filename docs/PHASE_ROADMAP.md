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

현재 확인 상태:

- Mount Integration Gate: PASS
- Natural Posture Gate: PASS
- Wrist Transition Gate: FAIL (`5.12rad/s > 2.0rad/s`)
- Forward Reach Gate: FAIL (`10.165mm > 10mm`, 금지 충돌 0건)
- Orientation Alignment Gate: FAIL (opt-in 경로 self-collision 잔존)
- Precontact Tracking Gate: 미도달 (Forward Reach 선행 Gate 실패)
- Gate A: FAIL (`0/30`), Gate B/C/D: 미시도

Gate A 이전 실패를 성공으로 포장하거나 threshold를 낮추지 않는다. Gate
A~D를 통과하기 전에 Factory/IL/BC/PPO를 구현하지 않는다.

## Final target

최종 목표는 G1이 실제 whole-body locomotion과 manipulation을 사용해
fault 발생 지점으로 이동하고, 자세를 조절하고, 복구하고, 완료를 검증하는
mission 전체다. Planar base는 개발·디버깅 fallback일 뿐 최종 목표를
대체하지 않는다.

Recovery skill마다 순서는 고정한다.

```text
Environment -> Expert -> Demonstrations -> BC -> Evaluation -> PPO
```
