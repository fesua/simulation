#!/usr/bin/env bash
# Exit 0 while an Isaac eval owns the GPU, 1 when it is free.
#
# NOT `pgrep -f "...eval_closed_loop"`: that matches ANY process whose command line merely
# CONTAINS the string, including the shell that wrote this script via a heredoc and the agent
# tool-call shells. Two queued jobs deadlocked on exactly that. Match the process NAME as python
# and require the script in its own argv instead.
ps -eo comm=,args= | awk '$1 ~ /^python/ && /scripts\/eval_closed_loop\.py/ { n++ } END { exit !n }'
