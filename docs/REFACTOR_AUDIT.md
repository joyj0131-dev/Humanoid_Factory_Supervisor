# Sharpa-only Renewal Audit

## Result

활성 branch는 bare G1 + Sharpa 단일 모델로 전환됐다. Dex3 env, controller,
planner, diagnostics, tests, viewer modes는 archive branch에만 남는다.

## Preserved contracts

| 영역 | 결과 |
|---|---|
| Foundation reach | observation 43 / action 14 유지, tests 17/17 |
| Scripted expert/data | tests 14/14 |
| BC/evaluation plumbing | tests 17/17 |
| Whole-body | Sharpa 8 hand groups로 action 37, tests 21/21 |
| Sharpa standalone | 9/9 |
| G1+Sharpa integration | 10/10, nq/nv/nu=80/79/73 |
| Hand demo | 4/4 |
| Bimanual grasp | 14/15 (유일한 실패는 실제 Gate A 미달성 assertion) |

## Removed active code

- Dex3 fixed-base env/controller and synergy module
- diagonal/manifold/tripod/force-trace planners and diagnostics
- 대응 실행·테스트 scripts
- single-hand Sharpa prototype
- viewer의 hand 선택 및 legacy diagnostic branches

삭제된 내용은 `phase4/dex3-grasp @ bce1dec`와 `phase4-dex3-end` tag에
보존된다.

## Model correction

이전 Sharpa builder는 pre-hand-equipped G1에서 손가락 body/geom만 지워
손목당 0.202839kg의 과거 hand inertia를 남겼다. 현재 builder는 bare
`g1.xml`에서 시작한다. `MjSpec.attach` 후 keyframe 순서가 바뀌는 문제는
joint/actuator name 기반 keyframe 복구로 해결했으며, 좌우 대칭·stand
collision 0·3초 wrist drift 7.394mm를 재검증했다.

## Intentionally retained local assets

`assets/robots/g1/`는 Git ignore된 upstream menagerie 설치 디렉터리다.
`g1_with_hands.xml`과 관련 mesh가 로컬 bundle에 존재할 수 있지만 active
Python/MJCF entry point에서 참조하지 않는다. 이를 부분 삭제하면 upstream
bundle 무결성과 archive branch 재현성이 깨지고 tracked source 단순화에는
효과가 없으므로 로컬 자산은 보존한다.
