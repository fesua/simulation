#!/usr/bin/env bash
# Side-by-side comparison of the same seeded episode under N policies.
# usage: compose_policy_compare.sh <out.mp4> <label1>=<video1> [<label2>=<video2> ...]
set -euo pipefail
out=$1; shift
inputs=(); filters=(); labels=(); i=0
for pair in "$@"; do
  label=${pair%%=*}; vid=${pair#*=}
  inputs+=(-i "$vid")
  filters+=("[$i:v]scale=640:480,drawbox=y=0:h=34:c=black@0.6:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='$label':x=10:y=6:fontsize=22:fontcolor=white[v$i];")
  labels+=("[v$i]")
  i=$((i+1))
done
ffmpeg -y -hide_banner -loglevel error "${inputs[@]}" \
  -filter_complex "$(printf '%s' "${filters[@]}")$(printf '%s' "${labels[@]}")hstack=inputs=$i[out]" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$out"
echo "wrote $out"
