---
name: sim-controller-fidelity-over-smoothing
description: Isaac 평가 리그에서 떨림은 제어기 스무딩이 아니라 물리 충실도로 잡는다는 운영자 원칙
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 28c2462f-f019-41d1-9d5e-862b2ebfe17f
---

`~/workspace/simulation` Isaac 평가 리그에서 **제어기 파라미터를 스무딩 쪽으로 조절해 떨림을
없애는 것은 금지**. 제어기는 모델이 낸 chunk를 최대한 그대로 추종해야 하고, 떨림은 sim의
물리를 실기에 맞춰서(real처럼) 잡는 것이 목표다.

**Why:** 이 리그의 1순위 목표는 정책의 task 성능 평가다. 명령을 필터링해 떨림을 가리면
평가 대상인 정책 출력이 왜곡되고, 특히 pi0.5는 proprio에 velocity가 들어가므로 제어기가
관측 자체를 오염시킨다 — (B) 충실형 컨트롤러 + chunk follower를 고른 이유가 바로 이것이다.
스무딩으로 맞춘 sim은 실기와의 상관(사다리 6단계)을 주장할 근거를 잃는다.

**How to apply:** 떨림 대책은 접촉/플랜트 모델 쪽에서 찾는다 — 강체 핑거 vs 실기의 AgileX
Fin-Ray 유연 핑거, 드라이브 게인, PhysX 솔버 반복/접촉 파라미터 등. 금지 대상은 crossfade,
smoothing_window 확대, corner_velocity_scale 같은 감쇠 가드, 명령 저역통과, speed scale 축소.
단 실기 `cartesian_chunk_follower.cpp:301`의 `vf = (d_k + d_kp1)/(2·dt)` 중앙차분 목표속도는
스무딩이 아니라 **충실도**다 — 이것이 없으면 33 ms knot마다 완전 정지해 오히려 chunk가
함의하는 연속 운동에서 벗어난다. 실측으로 knot 주파수 에너지 20.8% → 12.9%.

관련: [[cm-followunit-fidelity-gap]], [[isaac-sim-eval-rig-project]],
[[flow-infer-tremble-stall-diagnosis]]
