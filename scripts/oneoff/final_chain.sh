#!/usr/bin/env bash
# One-off driver for the M1b test-once runs (Task 6): 3B + adapters, then 7B, then 14B.
cd ~/code/sqltune
log=outputs/final_eval.log
echo "### chain start $(date +%H:%M)" >> $log
make --no-print-directory serve-down >> $log 2>&1
if make --no-print-directory serve-multi ADAPTERS="m0_qlora m1b_mix" SERVE_WAIT=900 >> $log 2>&1; then
  make --no-print-directory m1-eval GROUP=final3b >> $log 2>&1
else
  echo "### 3B DID NOT START" >> $log
fi
make --no-print-directory serve-down >> $log 2>&1
echo "### serve qwen7b $(date +%H:%M)" >> $log
if make --no-print-directory serve MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ SERVED_NAME=qwen7b SERVE_WAIT=900 >> $log 2>&1; then
  make --no-print-directory m1-eval GROUP=final7b >> $log 2>&1
else
  echo "### qwen7b DID NOT START" >> $log
fi
make --no-print-directory serve-down >> $log 2>&1
echo "### serve qwen14b $(date +%H:%M)" >> $log
if GPU_UTIL=0.88 make --no-print-directory serve MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ SERVED_NAME=qwen14b EXTRA_LORA=--enforce-eager SERVE_WAIT=900 >> $log 2>&1; then
  make --no-print-directory m1-eval GROUP=final14b >> $log 2>&1
else
  echo "### qwen14b DID NOT START" >> $log
  docker compose logs vllm 2>&1 | tail -30 >> $log
fi
make --no-print-directory serve-down >> $log 2>&1
uv run sqltune summarize >> $log 2>&1
echo "### chain done $(date +%H:%M)" >> $log
