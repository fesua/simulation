#!/usr/bin/env bash
# Score an ALREADY-LOCAL checkpoint on std40, waiting for the eval GPU to free up first.
# Same contract as chain_eval_va.sh minus the remote pull.
#   $1 config  $2 local model dir (containing params/ assets/)  $3 label  $4 port
set -uo pipefail
CFG=$1; DIR=$2; LABEL=$3; PORT=$4
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
LOG=/home/plaif/chain_eval.log
say() { echo "=== $(date -u +%H:%M:%S) $* ===" >> "$LOG"; }

# One scoring job at a time: sharing the GPU changes tick timing, and timing has already been
# shown to change outcomes on this rig.
while /home/plaif/workspace/simulation/scripts/eval_gpu_busy.sh; do sleep 20; done

slog=/home/plaif/serve_${LABEL}_${PORT}.log
say "SERVE $LABEL (local)"
( cd "$OPENPI" && XLA_PYTHON_CLIENT_PREALLOCATE=false exec .venv/bin/python scripts/serve_policy.py \
    --num-medoid-samples 1 --port "$PORT" policy:checkpoint \
    --policy.config "$CFG" --policy.dir "$DIR" ) > "$slog" 2>&1 &
spid=$!
ok=0
for _ in $(seq 1 150); do
  grep -q "server listening" "$slog" 2>/dev/null && { ok=1; break; }
  kill -0 $spid 2>/dev/null || break
  sleep 5
done
[ "$ok" -ne 1 ] && { say "SERVE FAILED $LABEL"; tail -6 "$slog" >> "$LOG"; kill $spid 2>/dev/null; exit 1; }

cd "$SIM" || exit 1
PROTOCOL=std40 scripts/run_std20.sh "$LABEL" 127.0.0.1 "$PORT" aligned \
    --rtc --scene-states assets/scene_states40_aligned.json
kill $spid 2>/dev/null
say "$LABEL EVAL DONE"
