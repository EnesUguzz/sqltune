#!/usr/bin/env bash
# One-off: is the 7B fine-tune's small BIRD gain an AWQ artefact? Serve the merged bf16 fine-tune
# and the bf16 base with vLLM online FP8 (8-bit weights, near lossless) and eval both on dev.
cd ~/code/sqltune
log=outputs/fp8check.log
run () {  # model, served name, config dir
  make --no-print-directory serve-down >> $log 2>&1
  echo "### serve $2 $(date +%H:%M)" >> $log
  if GPU_UTIL=0.88 make --no-print-directory serve MODEL=$1 SERVED_NAME=$2 EXTRA_LORA="--quantization fp8" SERVE_WAIT=900 >> $log 2>&1; then
    for f in experiments/configs/m1b/$3/*.yaml; do
      echo "=== $f $(date +%H:%M)" >> $log
      uv run sqltune eval --config $f 2>&1 | grep -E "^results|WARNING|Error" >> $log || echo "    FAILED $f" >> $log
    done
  else
    echo "### $2 DID NOT START" >> $log
    docker compose logs vllm 2>&1 | tail -30 >> $log
  fi
}
run /models/m1b_mix7b-bf16 m1b_mix7b_fp8 fp8check
run Qwen/Qwen2.5-Coder-7B-Instruct qwen7b_fp8 fp8check_base
make --no-print-directory serve-down >> $log 2>&1
echo "### fp8check done $(date +%H:%M)" >> $log
