#!/usr/bin/env bash
# Tier 0: questions answerable with checkpoints we already have. No training.
#   griponly  -- drop_velocity_proprio=True. Does removing velocity proprio change aim precision?
#                (caveat: 60k steps, not the 80k of the control -- not a perfect pair)
#   v2_nolang -- blank task sentence. Closed-loop confirmation on the fixed set.
# Both are pi05-family, so this is a PRECISION comparison inside the working regime,
# not a works/doesn't-work comparison.
set -uo pipefail
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
PORT=8011
LOG=/home/plaif/tier0.log
ARMS="griponly:pi05_pika_umi_wrist_griponly_k1_h24_80k:/home/plaif/workspace/pika_umi_models_v2/wrist_griponly_k1_80k/60000
v2nolang:pi05_pika_umi_wrist_velgrip_k1_h24_80k_v2_nolang:/home/plaif/workspace/pika_umi_models_v2/v2_nolang/79999"

echo "$ARMS" | while IFS=: read -r tag cfg dir; do
  [ -d "$dir" ] || { echo "=== SKIP $tag (no $dir) ===" | tee -a "$LOG"; continue; }
  slog=/home/plaif/serve_${tag}_${PORT}.log
  echo "=== $(date -u +%H:%M:%S) SERVE $tag ($cfg) ===" | tee -a "$LOG"
  ( cd "$OPENPI" && XLA_PYTHON_CLIENT_PREALLOCATE=false exec .venv/bin/python scripts/serve_policy.py \
      --num-medoid-samples 1 --port $PORT policy:checkpoint \
      --policy.config "$cfg" --policy.dir "$dir" ) > "$slog" 2>&1 &
  spid=$!
  ok=0
  for _ in $(seq 1 120); do
    grep -q "server listening" "$slog" 2>/dev/null && { ok=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "=== FAILED to serve $tag ===" | tee -a "$LOG"; tail -6 "$slog" | tee -a "$LOG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; continue
  fi
  "$SIM/scripts/run_std20.sh" "$tag" 127.0.0.1 $PORT aligned --rtc
  echo "=== $(date -u +%H:%M:%S) STOP $tag ===" | tee -a "$LOG"
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
done
echo "=== $(date -u +%H:%M:%S) TIER0 DONE ===" | tee -a "$LOG"
