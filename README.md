# Isaac Sim 폐쇄루프 평가 리그 — dual RB3-730E + Pika, M12 볼트 pick & place

양팔 RB3-730E + Pika 그리퍼 셀의 **M12 볼트 pick-place VLA(pi0.5/openpi)를 Isaac Sim에서
폐쇄루프로 채점**하는 리그입니다. 실기 rollout을 대체해 모델 실험을 가속하는 것이 목적이며,
실기 라벨이 있는 체크포인트들을 전부 올바른 순위로 갈라 **순위 판정자(rank judge)** 로
검증됐습니다 (절대 성공률은 실기와 비교하지 않습니다).

![oracle run](docs/img/oracle_run.png)

- 상세 핸드오프/이력: [`CLAUDE.md`](CLAUDE.md) (빌드 사다리, 함정 목록, 단계별 검증 기록)
- 연구 기록: llm-wiki `projects/isaac-sim-eval-rig.md`

## 검증 상태 (2026-08-23)

| 항목 | 상태 |
|---|---|
| FK 정합 (MuJoCo 교차검증) | 최대 0.00038 mm / 0.051° |
| 실기-시뮬 순위 상관 | 실기 라벨 체크포인트 5/5 올바른 쪽 |
| **파지 물리 (physics v1)** | 힘제한 스톨 + 조 지연 실측 이식. 제대로 도달한 close의 94–100% 파지, 조 간격이 M12 머리(18.4mm)에 정확히 스톨 |
| 호스트 동등성 | RTX 5090 ↔ RTX PRO 6000 1pp 이내 |

## 설치

Ubuntu 22.04 (glibc 2.35), NVIDIA Vulkan ICD 필요 (`/usr/share/vulkan/icd.d/` 또는
`/etc/vulkan/icd.d/`에 `nvidia_icd.json`).

```bash
uv venv --python 3.12 .venv-isaac
uv pip install --python .venv-isaac/bin/python --prerelease=allow \
  "isaacsim[all,extscache]==6.0.1.0" msgpack imageio imageio-ffmpeg ruckig
```

함정 (전부 실제로 밟은 것):
- `--prerelease=allow` 필수 (`tinyobjloader` rc 의존성)
- 첫 실행에 `OMNI_KIT_ACCEPT_EULA=YES`
- Python **3.12 고정** (6.x 요구사항), venv 용량 ~24GB
- 렌더 확인은 반드시 `rep.orchestrator.step()` 경유 — `world.step(render=True)`만으로는 애노테이터가 비어 있음

스모크 테스트 (부팅 + 물리 + 오프스크린 렌더):

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/smoke_test.py
```

![smoke](docs/img/smoke_test.png)

## 사용법

### 정책 평가 (std40 고정 프로토콜)

openpi `serve_policy.py` websocket 서버에 직접 접속해 30Hz 폐쇄루프로 돕니다.
배포 계약(H=24, execute 4, chunk_anchor=command, RTC, velproprio=command)을 그대로 따릅니다.

```bash
export PYTHONPATH=<openpi>/packages/openpi-client/src
PROTOCOL=std40 scripts/run_std20.sh <label> 127.0.0.1 <port> aligned \
    --rtc --scene-states assets/scene_states40_aligned.json
```

- **모든 모델은 같은 동결 씬으로만 채점** (`scene_states40_aligned.json`, seed 100–139).
  PhysX 정착이 비결정적이라 동결 없이는 같은 seed도 다른 세계가 됩니다.
- **비디오 ON 고정** — 끄면 orchestrator 스텝 수가 달라져 결과가 바뀝니다 (실측).
- 집계: `scripts/t1_report.py --table outputs/eval/summary_*.json`

### 오라클 (특권 상태 파지-물리 측정기)

정책 대신 스크립트 플래너가 정확한 볼트 pose로 pick & place — **상한선 측정과 물리
캘리브레이션 전용**입니다. 씬·컨트롤러·채점은 정책 평가와 동일합니다.

```bash
env GRIP_RATE=3 GRIP_MAXF=5 ORACLE_ARM=left GRIP_OPEN=50 BOLT_SPREAD=2.5 \
  .venv-isaac/bin/python scripts/eval_closed_loop.py --oracle \
  --layout aligned --episodes 6 --episode-sec 60 --n-per-color 10 --rtc \
  --tag my_run --scene-states assets/scene_states_sparse.json
```

### 물리/계측 노브 (환경변수)

| 노브 | 의미 | 적용 |
|---|---|---|
| `GRIP_MAXF` | 핑거 드라이브 힘 제한 [N] — 실기 모터 스톨 재현. **physics v1 = 5** | 정책+오라클 |
| `GRIP_LAG_L/R` | 조 수송 지연 [ms] — 실기 실측 **105/209** | 정책+오라클 |
| `GRIP_PROPRIO` | grip proprio 소스 `command`/`actual` — 배포 기본은 `actual` | 정책+오라클 |
| `GRIP_LEAD_L/R` | 청크 내 grip 채널을 pose 대비 N스텝 시프트 | 정책+오라클 |
| `BOLT_FRICTION` / `FINGER_FRICTION` | 접촉 재질 (미설정 시 PhysX 기본 0.5) | 정책+오라클 |
| `GRIP_RATE` | 닫힘 램프 [%/tick] — 오라클 플래너용 (정책은 자체 속도로 닫음) | 오라클 |
| `GRIP_OPEN` | 열림 상한 [%] — 50이면 밀집 더미에서 이웃 걸림 대폭 감소 | 오라클 |
| `ORACLE_ARM` | `left`/`right` 단팔 모드 (반대팔 q=0 주차) | 오라클 |
| `BOLT_SPREAD` | 더미 산포 배율 (희소 씬 생성용; 채점은 1.0 고정) | 씬 |

### 에피소드 샤딩 (한 실험을 GPU 여러 개로)

에피소드는 동결 씬에서 독립 리셋이므로 `--seed` 오프셋으로 쪼개면 통계적으로 동일합니다.
루프가 완전 동기식이라 **벽시계 속도·병렬 실행은 결과에 영향을 주지 않습니다.**

```bash
# GPU0: seeds 100–102, GPU1: seeds 103–105  →  분석 시 summary 두 개를 합침
CUDA_VISIBLE_DEVICES=0 ... --episodes 3 --seed 100 --tag exp_s0 &
CUDA_VISIBLE_DEVICES=1 ... --episodes 3 --seed 103 --tag exp_s1 &
```

### close 이벤트 실패 분해 지표

close 명령마다 기록: 조준(`dxy`,`dz`) / 조-볼트 각도(`jaw_bolt_deg`, 성공의 96%가 >60°) /
접근 기울기(`tcp_rotvec`) / 이웃 간섭(`neighbors_in_jaw`) / 볼트 얹힘(`bolt_elev_mm`) /
하강 막힘(`z_cmd_gap_mm`) / 실측 조 간격(`achieved_gap_mm`, 스톨 서명) / 사출(`bolt_move_mm`).
에피소드 단위: `lifted`/`dropped`로 **집기와 이송을 분리** 채점.

## 그리퍼 기하 검증

CAD(STEP)와 sim 메시의 팁 형상·간격 대조 도면은 `scripts/gripper_sections.py` /
`scripts/tips_gripper_vs_sense.py`로 생성합니다.

![gripper open vs closed](docs/img/overlay_open_vs_closed_xz.png)

## 저장소 구성

```
scripts/   평가·씬·오라클·분석 스크립트 (핵심: eval_closed_loop.py)
assets/    검증된 로봇 USD, 동결 씬 상태(json), 그리퍼 메시
docs/img/      README 이미지
docs/memory/   에이전트 핸드오프 메모 (.md)
CLAUDE.md  프로젝트 핸드오프 문서 (이력·함정·검증 기록 전체)
```
