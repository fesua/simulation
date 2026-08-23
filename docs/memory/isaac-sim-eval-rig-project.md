---
name: isaac-sim-eval-rig-project
description: "~/workspace/simulation — Isaac Sim 볼트 pick-place 평가 리그 프로젝트, 핸드오프 문서는 CLAUDE.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28c2462f-f019-41d1-9d5e-862b2ebfe17f
---

2026-08-17 시작. `~/workspace/simulation`에 dual RB3-730E + Pika M12 볼트 pick-place의
**Isaac Sim 폐쇄루프 평가 리그**를 구축하는 프로젝트. **1순위 = task 성능 평가**(무작위 배치,
nearest-bolt 일치/불일치 분리 집계), sim 학습데이터 생성은 의도적 후순위 (eval-first, Genesis 논지).

- 전체 맥락·구축 사다리·함정(스탠드 90° 회전 합성, reset pose 값, hand-eye unmeasured,
  meshdir/누락 asset 등)은 **`~/workspace/simulation/CLAUDE.md`가 소스오브트루스**.
- `montage/expected_sim_montage.png` = 사용자 승인된 목표 구도 스펙 (MuJoCo 프리뷰).
- 당시 상태: 로컬 :8001 pi0.5가 정렬 배치에서 80~90% 성공 (운영자 판정).
- 관련: [[pika-openpi-copycat-proprio-diagnosis]] (색상 조건화 unlearnable 판정이 평가 설계의 근거),
  [[gpu-train-servers-layout]] (대량 평가 시 이전 후보 서버).

**Why**: 실기 평가는 수동·저속(자동 판정기 없음)이라 무작위 배치 대량 평가의 병목.
**How to apply**: simulation/ 디렉토리 작업 시 CLAUDE.md의 사다리 단계와 게이트(특히
hand-eye 측정 선행, 정렬 배치 상관 확인 후 무작위 신뢰)를 따를 것.
