#!/usr/bin/env bash
# Wait until fewer than $1 Isaac eval instances are running (default 2).
# The eval loop is fully synchronous, so parallel instances cannot change sim results --
# wall-clock speed never leaks into sim ticks. GPU measured 14% busy with 20 GB free.
MAX=${1:-2}
while :; do
  n=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /scripts\/eval_closed_loop\.py/ {c++} END {print c+0}')
  [ "$n" -lt "$MAX" ] && exit 0
  sleep 20
done
