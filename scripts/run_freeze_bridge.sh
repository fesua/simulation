#!/usr/bin/env bash
# Waits for the running v2nolang score to finish, then:
#   1. applies the scene-freezing patch (never while a scoring queue is live)
#   2. captures the settled world for seeds 100-119, aligned and random
#   3. ACCEPTANCE TEST: same seed twice must now report an identical scene
#   4. scores pi05v1 on the frozen scenes -- the bridge between the two protocols
#
# Step 3 is not ceremony. If the scene still moves after freezing, the non-determinism is not in
# the settle and the whole premise is wrong; better to find that out before scoring anything.
set -uo pipefail
SIM=/home/plaif/workspace/simulation
LOG=/home/plaif/freeze_bridge.log
say() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a "$LOG"; }

cd "$SIM" || exit 1

say "waiting for v2nolang to finish"
while pgrep -f "eval_closed_loop.py --protocol std20" >/dev/null; do sleep 30; done
sleep 10

say "applying scene-freeze patch"
.venv-isaac/bin/python scripts/scene_freeze_patch.py 2>&1 | tee -a "$LOG"
.venv-isaac/bin/python -m py_compile scripts/eval_closed_loop.py || { say "COMPILE FAILED"; exit 1; }

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/plaif/workspace/openpi/packages/openpi-client/src
PY=.venv-isaac/bin/python

for lay in aligned random; do
  say "capturing settled scenes: $lay"
  timeout 3600 $PY scripts/eval_closed_loop.py --protocol std20 --layout $lay \
      --tag freeze_$lay --port 8001 \
      --dump-scene-states assets/scene_states_$lay.json 2>&1 | grep -E "\[freeze\]|Error|Traceback" | tee -a "$LOG"
done

say "ACCEPTANCE TEST: same seed twice on frozen scenes must match exactly"
for i in 1 2; do
  timeout 900 $PY scripts/eval_closed_loop.py --episodes 1 --episode-sec 2 --seed 100 \
      --layout aligned --no-video --tag acc$i --port 8001 \
      --scene-states assets/scene_states_aligned.json 2>&1 \
    | grep -E "crowding|frozen scenes" | tee -a "$LOG"
done

say "BRIDGE: pi05v1 on frozen scenes (std20 aligned)"
scripts/run_std20.sh pi05v1_frozen 127.0.0.1 8001 aligned --rtc \
    --scene-states assets/scene_states_aligned.json
say "FREEZE BRIDGE DONE"
