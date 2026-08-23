#!/usr/bin/env bash
# 5-policy grid (3 on top, 2 on the bottom) for one seeded episode.
# usage: compose_grid5.sh <out.mp4> <seed> v1=<f> v2=<f> nolang=<f> c2=<f> c2r2=<f>
#
# The bottom row is padded rather than filled with a lavfi `color` source: a color
# source has no end tied to the real inputs, and hstacking it produced an
# ever-growing output (380 MB and climbing before it was killed).
set -euo pipefail
out=$1; seed=$2; shift 2
declare -A V; for p in "$@"; do V[${p%%=*}]=${p#*=}; done
order=(v1 v2 nolang c2 c2r2)
declare -A NAME=( [v1]="pi05 v1 (baseline)" [v2]="pi05 v2 (with language)" \
                  [nolang]="pi05 v2_nolang (empty prompt)" [c2]="C2 c2" [c2r2]="C2 c2_r2" )
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ins=(); fil=""; i=0
for k in "${order[@]}"; do
  ins+=(-i "${V[$k]}")
  txt="${NAME[$k]}"; [ "$i" -eq 0 ] && txt="$txt   |   seed $seed"
  fil+="[$i:v]scale=480:360,drawbox=y=0:h=26:c=black@0.65:t=fill,"
  fil+="drawtext=fontfile=$F:text='$txt':x=6:y=4:fontsize=17:fontcolor=white[v$i];"
  i=$((i+1))
done
fil+="[v0][v1][v2]hstack=inputs=3[top];[v3][v4]hstack=inputs=2[b0];"
fil+="[b0]pad=1440:360:0:0:color=0x202020[bot];[top][bot]vstack=inputs=2[out]"
ffmpeg -y -hide_banner -loglevel error "${ins[@]}" -filter_complex "$fil" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$out"
