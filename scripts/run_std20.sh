#!/usr/bin/env bash
# THE fixed scoring run. Every model is scored on exactly this: 20 episodes, seeds 100-119,
# 30 s each, 20 bolts (10 per colour), no video. Nothing here is a per-run choice -- if an
# experiment needs a different eval, it is a different protocol and must be named as such.
#
#   [PROTOCOL=std20|std40] run_std20.sh <label> <host> <port> [aligned|random] [extra args...]
#
# PROTOCOL is an env var, not a positional flag, so it can never be silently overridden by
# a duplicate --protocol in the extra args (argparse would take the last one and the intent
# would depend on argument order).
#
# Examples
#   run_std20.sh pi05_v1       127.0.0.1 8001 aligned --rtc
#   run_std20.sh pi05_v1_rand  127.0.0.1 8001 random  --rtc
#   run_std20.sh wm            127.0.0.1 8005 aligned
#
# Writes outputs/eval/summary_<label>_<layout>.json and appends to ~/std20_eval.log.
set -uo pipefail
LABEL=${1:?label}; HOST=${2:?host}; PORT=${3:?port}; LAYOUT=${4:-aligned}; shift 4 2>/dev/null || shift $#
SIM=/home/plaif/workspace/simulation
LOG=/home/plaif/std20_eval.log

# openpi_client lives in the openpi tree, not in .venv-isaac
export PYTHONPATH=/home/plaif/workspace/openpi/packages/openpi-client/src
export OMNI_KIT_ACCEPT_EULA=YES

echo "=== $(date -u +%H:%M:%S) START $LABEL $HOST:$PORT $LAYOUT $* ===" | tee -a "$LOG"
cd "$SIM" || exit 1
.venv-isaac/bin/python scripts/eval_closed_loop.py \
    --protocol "${PROTOCOL:-std20}" --layout "$LAYOUT" --tag "$LABEL" --label "$LABEL" \
    --host "$HOST" --port "$PORT" "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== $(date -u +%H:%M:%S) EXIT $LABEL rc=$rc ===" | tee -a "$LOG"
[ "$rc" -eq 0 ] && .venv-isaac/bin/python scripts/t1_report.py \
    "outputs/eval/summary_${LABEL}_${LAYOUT}.json" | tee -a "$LOG"
exit "$rc"
