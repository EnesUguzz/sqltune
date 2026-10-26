#!/usr/bin/env bash
# One-off driver: pass@8 and majority voting (T 0.7) for base and fine-tune, 3B (bf16 + LoRA)
# and 7B (FP8), on bird_diag and the BIRD validation DBs, via scripts/pass_at_k.py.
cd "$(dirname "$0")/../.." && mkdir -p outputs
log=outputs/passk.log
E=experiments/configs
pk () {  # eval config, run prefix
  for s in birddiag birdval; do
    set_arg=config; [ $s = birdval ] && set_arg=birdval
    echo "=== $2 $s $(date +%H:%M)" >> $log
    uv run python scripts/pass_at_k.py --eval-config $1 --set $set_arg \
      --greedy-run m1_$2__zero__$s --run-name m1_$2__zero__${s}_k8 2>&1 | grep -E "_k8 \{|Error" >> $log
  done
}
echo "### start $(date +%H:%M)" >> $log
make --no-print-directory serve-down >> $log 2>&1
if make --no-print-directory serve-multi ADAPTERS="m1b_mix" SERVE_WAIT=900 >> $log 2>&1; then
  pk $E/m1/eval/m1b--base__zero__birddiag.yaml base
  pk $E/m1/eval/m1b--m1b_mix__zero__birddiag.yaml m1b_mix
fi
make --no-print-directory serve-down >> $log 2>&1
if GPU_UTIL=0.88 make --no-print-directory serve MODEL=/models/m1b_mix7b-bf16 SERVED_NAME=m1b_mix7b_fp8 EXTRA_LORA="--quantization fp8" SERVE_WAIT=900 >> $log 2>&1; then
  pk $E/m1b/fp8check/zero__birddiag.yaml m1b_mix7b_fp8
fi
make --no-print-directory serve-down >> $log 2>&1
if GPU_UTIL=0.88 make --no-print-directory serve MODEL=Qwen/Qwen2.5-Coder-7B-Instruct SERVED_NAME=qwen7b_fp8 EXTRA_LORA="--quantization fp8" SERVE_WAIT=900 >> $log 2>&1; then
  pk $E/m1b/fp8check_base/base_zero__birddiag.yaml qwen7b_fp8
fi
make --no-print-directory serve-down >> $log 2>&1
echo "### done $(date +%H:%M)" >> $log
