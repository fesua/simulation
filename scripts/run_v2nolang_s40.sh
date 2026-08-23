#!/usr/bin/env bash
# Re-measure v2nolang on std40, chained after the current queue.
#
# WHY: v2nolang is the only arm whose grasp-per-close is out of line (16.9% against ~8% for both
# pi05v1 and v2lang). v2lang landing on top of pi05v1 killed the "it is the v2 recipe" reading, so
# the remaining explanations are (a) removing the fixed task sentence genuinely helps, or (b) the
# 16.9% is single-run noise -- it was one 20-episode measurement and grasp-per-close has never had
# its noise band measured. std40 doubles the sample on a superset of the same seeds.
#
# If (a) survives this it is the cheapest improvement available: the checkpoint already exists.
set -uo pipefail
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
PORT=8013
LOG=/home/plaif/std40.log
CKPT=/home/plaif/workspace/pika_umi_models_v2/v2_nolang/79999
CFG=pi05_pika_umi_wrist_velgrip_k1_h24_80k_v2_nolang
say() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a "$LOG"; }

# Wait on the queue's own completion marker, not on a process name: `pgrep -f` matches the
# checking command itself and has already produced two false "still running" reports here.
say "v2nolang chain waiting for STD40 QUEUE DONE"
while ! grep -q "STD40 QUEUE DONE" "$LOG" 2>/dev/null; do sleep 60; done
sleep 30

cd "$SIM" || exit 1
slog=/home/plaif/serve_v2nolang_${PORT}.log
say "SERVE v2nolang ($CFG)"
( cd "$OPENPI" && XLA_PYTHON_CLIENT_PREALLOCATE=false exec .venv/bin/python scripts/serve_policy.py \
    --num-medoid-samples 1 --port $PORT policy:checkpoint \
    --policy.config "$CFG" --policy.dir "$CKPT" ) > "$slog" 2>&1 &
spid=$!
ok=0
for _ in $(seq 1 150); do
  grep -q "server listening" "$slog" 2>/dev/null && { ok=1; break; }
  kill -0 $spid 2>/dev/null || break
  sleep 5
done
if [ "$ok" -ne 1 ]; then
  say "FAILED to serve v2nolang"; tail -6 "$slog" | tee -a "$LOG"
  kill $spid 2>/dev/null; exit 1
fi

PROTOCOL=std40 "$SIM/scripts/run_std20.sh" v2nolang_s40 127.0.0.1 $PORT aligned \
    --rtc --scene-states assets/scene_states40_aligned.json
say "STOP v2nolang"
kill $spid 2>/dev/null
say "V2NOLANG S40 DONE"
