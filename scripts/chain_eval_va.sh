#!/usr/bin/env bash
# Wait for a remote training run to write its final checkpoint, pull it, and score it on std40.
# openpi writes the last checkpoint at N-1, never at N.
#   $1 host  $2 config  $3 exp tag  $4 local name  $5 step (39999/79999)  $6 port
set -uo pipefail
HOST=$1; CFG=$2; TAG=$3; NAME=$4; STEP=$5; PORT=$6
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
M=/home/plaif/workspace/pika_umi_models_v2
LOG=/home/plaif/chain_eval.log
say() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a "$LOG"; }

REMOTE=/home/plaif/workspace/openpi_runs/checkpoints/$CFG/$TAG/$STEP
say "waiting for $HOST:$REMOTE"
until timeout 30 ssh -o BatchMode=yes "$HOST" "test -d $REMOTE/params" 2>/dev/null; do sleep 300; done
sleep 120                                  # let the save finalise before copying

say "pulling $NAME"
mkdir -p "$M/$NAME/$STEP"
rsync -a "$HOST:$REMOTE/params" "$HOST:$REMOTE/assets" "$M/$NAME/$STEP/" || { say "PULL FAILED"; exit 1; }

# The local eval box runs one scoring job at a time: sharing the GPU changes tick timing, and
# timing has already been shown to change outcomes here.
while /home/plaif/workspace/simulation/scripts/eval_gpu_busy.sh; do sleep 20; done

slog=/home/plaif/serve_${NAME}_${PORT}.log
say "SERVE $NAME"
( cd "$OPENPI" && XLA_PYTHON_CLIENT_PREALLOCATE=false exec .venv/bin/python scripts/serve_policy.py \
    --num-medoid-samples 1 --port "$PORT" policy:checkpoint \
    --policy.config "$CFG" --policy.dir "$M/$NAME/$STEP" ) > "$slog" 2>&1 &
spid=$!
ok=0
for _ in $(seq 1 150); do
  grep -q "server listening" "$slog" 2>/dev/null && { ok=1; break; }
  kill -0 $spid 2>/dev/null || break
  sleep 5
done
[ "$ok" -ne 1 ] && { say "SERVE FAILED $NAME"; tail -6 "$slog" | tee -a "$LOG"; kill $spid 2>/dev/null; exit 1; }

cd "$SIM" || exit 1
PROTOCOL=std40 scripts/run_std20.sh "${NAME}_s40" 127.0.0.1 "$PORT" aligned \
    --rtc --scene-states assets/scene_states40_aligned.json
kill $spid 2>/dev/null
say "$NAME EVAL DONE"
