#!/usr/bin/env bash
# One-off audit driver, run sequentially (one GPU job at a time):
# 1. validation loss, base vs fine-tune, 3B and 7B (scripts/val_loss.py)
# 2. zero-shot EX on BIRD validation DBs and 250 BIRD training questions, base vs fine-tune,
#    3B (bf16 + LoRA) and 7B (FP8), via scripts/eval_in_distribution.py
# 3. base 3B with the SQLite-rules system prompt on dev (experiments/configs/m1b/sqlite_rules)
cd "$(dirname "$0")/../.." && mkdir -p outputs
log=outputs/indist.log
E=experiments/configs
echo "### start $(date +%H:%M)" >> $log
make --no-print-directory serve-down >> $log 2>&1
for m in mix:m1b_mix mix7b:m1b_mix7b; do uv run python scripts/val_loss.py $E/m1b/train/${m%%:*}.yaml adapters/${m#*:} 2>&1 | grep -E "val loss|Error" >> $log; done
ev () {  # eval config, run name prefix
  for s in val train; do
    echo "=== $2 $s $(date +%H:%M)" >> $log
    uv run python scripts/eval_in_distribution.py --eval-config $1 --set $s --run-name m1_$2__zero__bird$s 2>&1 | grep -E "questions on|results/" >> $log
  done
}
make --no-print-directory serve-down >> $log 2>&1
if make --no-print-directory serve-multi ADAPTERS="m1b_mix" SERVE_WAIT=900 >> $log 2>&1; then
  ev $E/m1/eval/m1b--base__zero__birddiag.yaml base
  ev $E/m1/eval/m1b--m1b_mix__zero__birddiag.yaml m1b_mix
  for f in $E/m1b/sqlite_rules/*.yaml; do
    echo "=== $f $(date +%H:%M)" >> $log
    uv run sqltune eval --config $f 2>&1 | grep -E "^results|Error" >> $log || echo "    FAILED $f" >> $log
  done
fi
make --no-print-directory serve-down >> $log 2>&1
if GPU_UTIL=0.88 make --no-print-directory serve MODEL=/models/m1b_mix7b-bf16 SERVED_NAME=m1b_mix7b_fp8 EXTRA_LORA="--quantization fp8" SERVE_WAIT=900 >> $log 2>&1; then
  ev $E/m1b/fp8check/zero__birddiag.yaml m1b_mix7b_fp8
fi
make --no-print-directory serve-down >> $log 2>&1
if GPU_UTIL=0.88 make --no-print-directory serve MODEL=Qwen/Qwen2.5-Coder-7B-Instruct SERVED_NAME=qwen7b_fp8 EXTRA_LORA="--quantization fp8" SERVE_WAIT=900 >> $log 2>&1; then
  ev $E/m1b/fp8check_base/base_zero__birddiag.yaml qwen7b_fp8
fi
make --no-print-directory serve-down >> $log 2>&1
echo "### done $(date +%H:%M)" >> $log
