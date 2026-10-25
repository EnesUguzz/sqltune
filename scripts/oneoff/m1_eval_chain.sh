#!/usr/bin/env bash
# One-off driver for the M1a eval matrix (server for 3B+adapters is already up).
cd ~/code/sqltune
log=outputs/m1_eval.log
echo "### chain start $(date +%H:%M)" >> $log
make --no-print-directory m1-eval GROUP=base3b >> $log 2>&1
make --no-print-directory m1-eval GROUP=adapters >> $log 2>&1
make --no-print-directory serve-down >> $log 2>&1
for pair in "qwen7b:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" "qwen14b:Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"; do
  name=${pair%%:*}; model=${pair#*:}
  echo "### serve $name $(date +%H:%M)" >> $log
  if make --no-print-directory serve MODEL=$model SERVED_NAME=$name SERVE_WAIT=900 >> $log 2>&1 \
     || { make --no-print-directory serve-down >> $log 2>&1; \
          echo "### retry $name with MAX_MODEL_LEN=6144 GPU_UTIL=0.90" >> $log; \
          MAX_MODEL_LEN=6144 GPU_UTIL=0.90 make --no-print-directory serve MODEL=$model SERVED_NAME=$name SERVE_WAIT=900 >> $log 2>&1; }; then
    make --no-print-directory m1-eval GROUP=$name >> $log 2>&1
  else
    echo "### $name DID NOT FIT" >> $log
    docker compose logs vllm 2>&1 | tail -30 >> $log
  fi
  make --no-print-directory serve-down >> $log 2>&1
done
uv run sqltune summarize >> $log 2>&1
echo "### chain done $(date +%H:%M)" >> $log
