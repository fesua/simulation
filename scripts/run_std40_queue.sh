#!/usr/bin/env bash
# std40 queue: regenerate the frozen world for seeds 100-139, then score the waiting checkpoints.
#
# Order is deliberate. `pad` is the paired CONTROL for three separate arms (nopad, griploss,
# res384), so it is scored first -- if the queue is interrupted, the control is the one result
# that must exist. pi05v1 is last because it is a reference point, not a comparison partner.
#
#   pad      40k, resize_with_pad          <- control for the 40k family
#   nopad    40k, resize_no_pad            <- resolution A/B partner
#   v2_lang  80k, v2 recipe WITH language  <- decomposes v2nolang's gain: recipe or language?
#   pi05v1   80k, v1 recipe                <- production reference on the new protocol
set -uo pipefail
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
LOG=/home/plaif/std40.log
PORT=8012
M=/home/plaif/workspace/pika_umi_models_v2
say() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a "$LOG"; }

cd "$SIM" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/plaif/workspace/openpi/packages/openpi-client/src
PY=.venv-isaac/bin/python

for lay in aligned random; do
  say "freezing std40 scenes: $lay (seeds 100-139)"
  timeout 5400 $PY scripts/eval_closed_loop.py --protocol std40 --layout $lay \
      --tag freeze40_$lay --port 8001 \
      --dump-scene-states assets/scene_states40_$lay.json 2>&1 \
    | grep -E "\[freeze\] wrote|Error|Traceback" | tee -a "$LOG"
done

ARMS="pad:pi05_pika_umi_wrist_velgrip_k1_h24_40k_pad:$M/pad_40k/39999
nopad:pi05_pika_umi_wrist_velgrip_k1_h24_40k_nopad:$M/nopad_40k/39999
v2lang:pi05_pika_umi_wrist_velgrip_k1_h24_80k_v2:$M/v2_lang/79999"

echo "$ARMS" | while IFS=: read -r tag cfg dir; do
  [ -d "$dir" ] || { say "SKIP $tag (missing $dir)"; continue; }
  slog=/home/plaif/serve_${tag}_${PORT}.log
  say "SERVE $tag ($cfg)"
  ( cd "$OPENPI" && XLA_PYTHON_CLIENT_PREALLOCATE=false exec .venv/bin/python scripts/serve_policy.py \
      --num-medoid-samples 1 --port $PORT policy:checkpoint \
      --policy.config "$cfg" --policy.dir "$dir" ) > "$slog" 2>&1 &
  spid=$!
  ok=0
  for _ in $(seq 1 150); do
    grep -q "server listening" "$slog" 2>/dev/null && { ok=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    say "FAILED to serve $tag"; tail -6 "$slog" | tee -a "$LOG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; continue
  fi
  PROTOCOL=std40 "$SIM/scripts/run_std20.sh" "${tag}_s40" 127.0.0.1 $PORT aligned \
      --rtc --scene-states assets/scene_states40_aligned.json
  say "STOP $tag"
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
done

say "SERVE pi05v1 (already on :8001)"
PROTOCOL=std40 "$SIM/scripts/run_std20.sh" pi05v1_s40 127.0.0.1 8001 aligned \
    --rtc --scene-states assets/scene_states40_aligned.json
say "STD40 QUEUE DONE"
