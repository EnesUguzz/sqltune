#!/usr/bin/env bash
# One-off overnight driver: after m1b_mix7b training, merge+AWQ the fine-tune and the base 7B
# (same script, same calibration), then dev evals of both and the official 7B AWQ rows3retry.
cd "$(dirname "$0")/../.." && mkdir -p outputs
log=outputs/chain7b.log
echo "### chain7b start $(date +%H:%M)" >> $log
while pgrep -f "[s]qltune train" > /dev/null; do sleep 60; done
if [ ! -f adapters/m1b_mix7b/adapter_config.json ]; then
  echo "### no adapter, training failed; stop" >> $log; exit 1
fi
echo "### training done $(date +%H:%M)" >> $log
PY=~/venvs/awq/bin/python
CAL=outputs/calib/m1b_mix7b.jsonl
[ -f models/m1b_mix7b-awq/config.json ] || $PY scripts/merge_awq.py --adapter adapters/m1b_mix7b --calib $CAL --out models/m1b_mix7b-awq >> outputs/awq7b.log 2>&1
echo "### awq fine-tune exit $? $(date +%H:%M)" >> $log
[ -f models/qwen7b-awq/config.json ] || $PY scripts/merge_awq.py --calib $CAL --out models/qwen7b-awq >> outputs/awq7b_base.log 2>&1
echo "### awq base exit $? $(date +%H:%M)" >> $log

serve_eval () {  # model path, served name, eval group
  make --no-print-directory serve-down >> $log 2>&1
  echo "### serve $2 $(date +%H:%M)" >> $log
  if make --no-print-directory serve MODEL=$1 SERVED_NAME=$2 SERVE_WAIT=900 >> $log 2>&1; then
    make --no-print-directory m1-eval GROUP=$3 >> $log 2>&1
  else
    echo "### $2 DID NOT START" >> $log
    docker compose logs vllm 2>&1 | tail -30 >> $log
  fi
}
[ -f models/m1b_mix7b-awq/config.json ] && serve_eval /models/m1b_mix7b-awq m1b_mix7b m1b7b
[ -f models/qwen7b-awq/config.json ] && serve_eval /models/qwen7b-awq qwen7b_own qwen7b_own
serve_eval Qwen/Qwen2.5-Coder-7B-Instruct-AWQ qwen7b qwen7b_rows
make --no-print-directory serve-down >> $log 2>&1
uv run sqltune summarize >> $log 2>&1
echo "### chain7b done $(date +%H:%M)" >> $log
