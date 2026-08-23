#!/usr/bin/env bash
# Render the SAME scene under several physics settings with the pi0.5 control policy, so the
# behaviour can be judged by eye instead of only through a success rate.
#
# Each run: one fixed frozen scene (seed picked by SEEDIDX), both the existing oblique overview
# and a near-top view side by side, 30 s. Output goes to simulation/physics/ and the clips are
# then stacked into a single labelled mp4.
#
#   scripts/physics_montage.sh [port] [seed-index]
set -uo pipefail
PORT=${1:-8001}
EPI=${2:-0}                      # index into the frozen std40 set (0 = seed 100)
SIM=/home/plaif/workspace/simulation
OUT=$SIM/physics
mkdir -p "$OUT"
cd "$SIM" || exit 1
export PYTHONPATH=/home/plaif/workspace/openpi/packages/openpi-client/src
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 EVAL_TOPVIEW=1

# name | env overrides.  "current" is the setting every published number was measured with.
CONFIGS=(
  "current|"
  "softfinger_kp1e4|GRIP_KP=1e4"
  "verysoft_kp1e3|GRIP_KP=1e3"
  "friction_hi_mu1.2|BOLT_FRICTION=1.2"
  "friction_lo_mu0.2|BOLT_FRICTION=0.2"
  "soft_and_grippy|GRIP_KP=1e4 BOLT_FRICTION=1.2"
)

# Wait for the eval GPU: sharing it changes tick timing, and timing changes outcomes here.
while /home/plaif/workspace/simulation/scripts/eval_gpu_busy.sh; do sleep 20; done

for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"; envs="${cfg#*|}"
  echo "=== $(date -u +%H:%M:%S) $name  [$envs] ==="
  # shellcheck disable=SC2086
  env $envs timeout 1200 .venv-isaac/bin/python -u scripts/eval_closed_loop.py \
      --layout aligned --episodes 1 --seed $((100 + EPI)) --episode-sec 30 \
      --n-per-color 10 --rtc --host 127.0.0.1 --port "$PORT" \
      --tag "phys_${name}" --label "phys_${name}" \
      --scene-states assets/scene_states40_aligned.json \
      > "$OUT/${name}.log" 2>&1
  rc=$?
  src="$SIM/outputs/eval/phys_${name}_aligned_00.mp4"
  [ -f "$src" ] && mv "$src" "$OUT/${name}.mp4"
  echo "=== $(date -u +%H:%M:%S) $name rc=$rc ==="
done
echo "=== CLIPS DONE ==="
