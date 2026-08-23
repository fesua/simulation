---
name: std20-eval-protocol
description: 모든 VLA/VA 모델을 채점하는 고정 평가 세트 std20 + T1 multimodality 계측 필드 (Isaac sim)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8cd91b78-5a92-55d8-af51-e0e038909174
  modified: 2026-08-19T15:35:25.351Z
---

2026-08-19 사용자 결정: **앞으로 모든 모델 실험·평가는 하나의 고정 20-에피소드 세트로 채점한다.**

`~/workspace/simulation/scripts/eval_closed_loop.py --protocol std20`
= 20 에피소드 / **seed 100~119** / 30초 / 볼트 20개(색당 10) / **비디오 ON**.
프로토콜이 이 노브들을 강제로 덮어써서 두 arm이 평가 설정 차이로 갈릴 수 없게 만든다.
실행은 `scripts/run_std20.sh <label> <host> <port> [aligned|random] [--rtc ...]`,
집계는 `scripts/t1_report.py summary_<label>_<layout>.json`.

**비디오 ON이 프로토콜인 이유 (2026-08-19, 대가를 치르고 배움)**: 끄면 2배 빠르지만
**결과가 달라진다.** 같은 코드·seed 104에서 ON은 과거 유효 런을 재현(correct 2, close dxy p50
10.6mm)했고 OFF는 벗어났다(correct 0, 260mm). 손목캠 관측 자체는 OFF에서도 정상임을 덤프로
확인했는데도 그렇다 — 녹화 여부가 `rep.orchestrator.step()` 호출 수를 바꾸고 orchestrator
스텝은 타임라인을 진행시킨다. **검증된 수치는 전부 비디오 ON에서 나왔다. 속도로 평가 기반을
바꾸지 말 것.** arm당 ~80분(30초 에피소드 ~4분 × 20).

**그 전에 밟은 지뢰**: `rep.orchestrator.step()`이 `if args.video:` 안에만 있어서 비디오를 끄면
손목캠 애노테이터가 안 채워지고 `_observe()`가 그걸 조용히 검은 이미지로 대체했다 →
10 에피소드가 0/0으로 채점되고 정책 결과처럼 보였다. 지금은 매 틱 항상 스텝하고, 빈 프레임은
`blank_obs`로 계수·첫 관측이 비면 abort·summary에 `!! INVALID RUN`. 관측 경로 감사는
`EVAL_DUMP_OBS=<dir>`로 정책이 받는 프레임을 직접 덤프해서 볼 것 — 집계 점수로는 감사 불가.

**T1 계측 필드** (close event마다, sim만 가능 — 모든 볼트의 ground truth pose를 알기 때문):
- `margin` = d(2nd nearest) − d(nearest) — 모호성 폭. 판정축은 **grasp 성공률 vs margin 곡선**:
  평평하면 multimodality는 범인이 아니고 정밀도/그립이 범인, 작은 margin에서 꺾이면 모드 문제.
- `t_seg`/`perp` — 조준점이 두 후보를 잇는 선분 위에 있는가 = **모드 평균 서명**
- `switches_1s`/`commit_lead_ticks` — 청크 경계 타깃 갈아타기 = **모드 스위치 서명**
- `nearest_is_target_color` — 무작위 배치에서만 의미 (색 조건화는 데이터상 unlearnable)
- 에피소드마다 `same_color_crowding_m`(동색 최근접 이웃 간격, **정착 직후 시작 씬** 기준) —
  쉬운 씬과 동률 씬을 같은 숫자로 묶으면 실험이 성립하지 않는다.

**Why**: 오프라인 지표는 이 프로젝트에서 세 번 연속 폐루프를 오판했고(l2 모드붕괴 / grip echo /
transit 속도), 실기 평가는 수동이라 느리다. 고정 폐루프 세트만이 비교 가능한 판정자다.
**How to apply**: 새 체크포인트를 판정할 때 자체 평가를 만들지 말고 std20을 그대로 돌릴 것.
다른 평가가 필요하면 그건 **다른 프로토콜**이고 이름을 새로 붙여야 한다.

관련: [[isaac-sim-eval-rig-project]], [[llm-removal-ladder-c2-verdict]], [[gpu-train-servers-layout]]
