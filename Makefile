CONFIG ?= experiments/configs/base.yaml
MODEL ?= Qwen/Qwen2.5-Coder-3B-Instruct
LORA ?=
# Seconds to wait for vLLM to report healthy (first start downloads the model).
SERVE_WAIT ?= 900

.PHONY: setup test lint format check-env serve serve-down download eval train m1-train serve-multi m1-eval summarize dataset-review demo

setup:
	uv sync --extra train

test:
	uv run pytest

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

check-env:  ## torch/CUDA/bitsandbytes/Unsloth + Docker GPU
	uv run sqltune check-env

serve:  ## vLLM on :8000; `make serve LORA=m0_qlora` also serves adapters/m0_qlora as model "sft"
	MODEL=$(MODEL) SERVED_NAME=$(SERVED_NAME) LORA_ARGS="$(if $(LORA),--lora-modules sft=/adapters/$(LORA),) $(EXTRA_LORA)" docker compose up -d vllm
	@for i in $$(seq 1 $(SERVE_WAIT)); do \
	  curl -fsS localhost:8000/health >/dev/null 2>&1 && { echo "vLLM ready"; curl -s localhost:8000/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"; exit 0; }; \
	  sleep 1; \
	done; echo "vLLM not healthy after $(SERVE_WAIT)s; see: docker compose logs vllm"; exit 1

serve-down:
	docker compose down

download:  ## Spider, BIRD Mini-Dev and BIRD train (questions + SQLite DBs) -> data/
	uv run sqltune download

eval:  ## score a model on Spider: make eval CONFIG=experiments/configs/m0/baseline.yaml
	uv run sqltune eval --config $(CONFIG)

train:  ## fine-tune: make train CONFIG=experiments/configs/m0/qlora.yaml (GPU must be free of vLLM)
	@if curl -fsS localhost:8000/health >/dev/null 2>&1; then echo "vLLM is running and holds the GPU; run make serve-down first"; exit 1; fi
	uv run --extra train sqltune train --config $(CONFIG)

M1_TRAIN ?= lc_f0125 lc_f025 lc_f05 lc_f1 others ep3 lora16 seed2 seed3

dataset-review:  ## gold health, duplicates and SQL mix of every set -> docs/data/dataset-review.md (+ BIRD train gold_failures.json)
	uv run python scripts/dataset_review.py

train-lengths:  ## token lengths of training records per source, rows 0/3, drops per max_seq_length
	uv run --extra train python scripts/measure_train_lengths.py

m1-train:  ## M1a training sweep, sequential; a failed run is logged and skipped (~3 h GPU)
	@mkdir -p outputs
	@for r in $(M1_TRAIN); do \
	  echo "=== $$r $$(date +%H:%M)"; \
	  $(MAKE) --no-print-directory train CONFIG=experiments/configs/m1/train/$$r.yaml > outputs/$$r.log 2>&1 \
	    && echo "    ok $$(date +%H:%M)" || echo "    FAILED (see outputs/$$r.log)"; \
	done

SERVED_NAME ?= base
ADAPTERS ?= m0_qlora lc_f0125 lc_f025 lc_f05 lc_f1 others ep3 lora16 seed2 seed3 m1b_mix

serve-multi:  ## base model + every adapter in ADAPTERS, each served under its own name
	$(MAKE) --no-print-directory serve LORA= SERVED_NAME=$(SERVED_NAME) \
	  EXTRA_LORA="--lora-modules $(foreach a,$(ADAPTERS),$(a)=/adapters/$(a)) --max-loras 4 --max-cpu-loras 16"

GROUP ?= base3b

m1-eval:  ## run every eval config of GROUP against the running server; failures logged, not fatal
	@mkdir -p outputs
	@for f in experiments/configs/m1/eval/$(GROUP)--*.yaml; do \
	  echo "=== $$f $$(date +%H:%M)"; \
	  uv run sqltune eval --config $$f 2>&1 | grep -E "^results|WARNING|Error" || echo "    FAILED $$f"; \
	done

demo:  ## Gradio app on :7860 against the running vLLM server (make serve-multi ADAPTERS=m1b_mix)
	uv run --extra demo python -m sqltune.demo

summarize:  ## one table over every results/m1_* run -> results/SUMMARY.md
	uv run sqltune summarize --out results/SUMMARY.md
