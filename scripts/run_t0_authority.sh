#!/usr/bin/env bash
# T0: does the sim RANK policies the way the real robot did?
#
# Every arm below already has a real-robot verdict on record (llm-wiki
# projects/vla-rollout-diagnosis.md). Scoring them on the SAME fixed set (std20) and comparing
# the order is the cheapest possible authority argument: no robot time, no calibration work.
# Absolute sim success does NOT have to match the robot for this to be useful -- rank does.
#
#   arm         real-robot verdict                                     served as
#   pi05v1      works, 80-90% operator-judged                          :8001 (already scored)
#   siglip      descends accurately, NEVER completes a pick            flow
#   ema999      aims between bolts, fails                              l2  (the branch deployed)
#   sigtok      fails                                                  flow
#   gripdr_ph   fails                                                  flow
#   wm          no real rollout -- paired control for siglip           flow
#   c2          no real rollout; sim 0/5 and worst offline             flow
#
# One arm at a time: the wrist renders and the policy share one GPU, and a second server would
# change the timing of the run it is being compared against.
set -uo pipefail
SIM=/home/plaif/workspace/simulation
OPENPI=/home/plaif/workspace/openpi
PORT=8010
LOG=/home/plaif/t0_authority.log
LAYOUT=${LAYOUT:-aligned}
ARMS=${ARMS:-"siglip:flow ema999:l2 wm:flow gripdr_ph:flow sigtok:flow c2:flow"}

for spec in $ARMS; do
  tag=${spec%%:*}; branch=${spec##*:}
  ck=/home/plaif/va_runs/$tag/step_80000.pt
  [ -f "$ck" ] || { echo "=== SKIP $tag (no checkpoint) ===" | tee -a "$LOG"; continue; }

  slog=/home/plaif/serve_${tag}_${PORT}.log
  echo "=== $(date -u +%H:%M:%S) SERVE $tag branch=$branch :$PORT ===" | tee -a "$LOG"
  ( cd "$OPENPI" && exec .venv/bin/python examples/pika_umi/serve_va.py \
      --ckpt "$ck" --branch "$branch" --port "$PORT" ) > "$slog" 2>&1 &
  spid=$!

  # Wait for the server to say it is serving, or die trying. Never assume it came up.
  ok=0
  for _ in $(seq 1 120); do
    grep -q "serving .* on " "$slog" 2>/dev/null && { ok=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "=== FAILED to serve $tag (see $slog) ===" | tee -a "$LOG"
    tail -5 "$slog" | tee -a "$LOG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null
    continue
  fi

  "$SIM/scripts/run_std20.sh" "$tag" 127.0.0.1 "$PORT" "$LAYOUT" --rtc
  echo "=== $(date -u +%H:%M:%S) STOP $tag ===" | tee -a "$LOG"
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  sleep 5
done
echo "=== $(date -u +%H:%M:%S) T0 QUEUE DONE ($LAYOUT) ===" | tee -a "$LOG"
