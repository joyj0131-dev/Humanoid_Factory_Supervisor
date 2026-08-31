# End-Effector: Sharpa Wave (Phase 4 Grasp Track)

> 상태: **부분 완료.** Stage 0(모델 감사/vendoring)과 Stage 1(단독 모델
> 검증)만 완료됐다. G1 장착, action mapping, controller, Gate 테스트,
> Dex3 비교는 아직 실행되지 않았다 — 이 문서의 해당 절은 그렇게 명시한다.
> 상세 세션 기록: `docs/history/PHASE4_GRASP_SESSION_35.md`(로컬 전용,
> `.gitignore`).

## 선택 이유

기존 Unitree Dex3-1(3-finger)에서 반복적으로(1~34차 세션) 관측된 엄지
충돌, edge/corner 접촉, force spike, 물체 회전/이탈 문제를 해결하기
위해, 사용자가 2026-08-31(35차 세션)에 end-effector를 5-finger 22-DoF
Sharpa Wave로 전환하기로 명시적으로 승인했다. G1 본체(어깨/팔꿈치/
손목/허리/다리)는 변경하지 않는다. Dex3-1은 삭제하지 않고 legacy
comparison baseline으로 보존한다.

## Upstream / License

- Repository: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
- Pinned commit: `6eea427eb24189519f32b9f21674cd534d3f973c`
- License: Apache License 2.0 (verbatim 보존: `assets/robots/sharpa_wave/LICENSE.txt`,
  `NOTICE.txt`)
- 이 프로젝트에 vendoring된 방식: `assets/robots/sharpa_wave/README.md`
  참고. ~23MB의 MJCF/mesh 서브셋은 git 이력에 직접 커밋하지 않고,
  `scripts/install_sharpa_wave_assets.py`가 pinned commit + checksum
  검증으로 재현한다(35차 세션에 클린 상태에서 재현성 실측 확인:
  재다운로드 결과가 원본과 byte-identical).

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

## G1 mounting transform

**미완료.** Stage 2(G1+Sharpa 통합)가 아직 실행되지 않았다. 이 절은
실제 컴파일된 통합 모델이 있을 때만 mount transform 수치로 채운다 —
지금 추측 기재하지 않는다.

## Action mapping / 전체 G1+Sharpa 차원

**미완료.** G1+Sharpa 통합 모델을 컴파일하기 전에는 총 actuator/
observation/action 차원을 문서에 기재하지 않는다(PROJECT_CONTEXT.md
Fixed Decisions에도 동일 원칙 적용).

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

## Dex3 비교 방법

**미완료.** 첫 SIZE_12(12cm 정육면체, half=0.06m) canonical grasp 비교는
Sharpa fixed-base Gate A를 통과한 뒤에만 실행한다(사용자 지시 Stage 5/6).
비교 표 형식은 사용자 지시의 Stage 6 지표 목록을 그대로 따른다.

## Viewer / 테스트 명령 (현재 실제로 존재하는 것만)

```
# Sharpa Wave 손 모델 감사 (읽기 전용, 컴파일된 MjModel 정보 출력)
python3 scripts/audit_sharpa_wave.py

# Sharpa Wave 단독 모델 검증 (9개 테스트)
python3 scripts/test_sharpa_wave_model.py

# Sharpa Wave 자산 설치/재현 (클론 직후 1회)
python3 scripts/install_sharpa_wave_assets.py
```

`--hand-model sharpa` 같은 viewer 플래그, G1+Sharpa 통합 viewer,
grasp controller 관련 명령은 **아직 구현되지 않았다** — Stage 2/4/7이
완료된 뒤에 이 절을 갱신한다.
