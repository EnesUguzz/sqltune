CONFIG ?= experiments/configs/base.yaml
MODEL ?= Qwen/Qwen2.5-Coder-3B-Instruct
LORA ?=
# Seconds to wait for vLLM to report healthy (first start downloads the model).
SERVE_WAIT ?= 900

.PHONY: setup test lint format check-env serve serve-down download eval train m1-train

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
	MODEL=$(MODEL) LORA_ARGS="$(if $(LORA),--lora-modules sft=/adapters/$(LORA),)" docker compose up -d vllm
	@for i in $$(seq 1 $(SERVE_WAIT)); do \
	  curl -fsS localhost:8000/health >/dev/null 2>&1 && { echo "vLLM ready"; curl -s localhost:8000/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"; exit 0; }; \
	  sleep 1; \
	done; echo "vLLM not healthy after $(SERVE_WAIT)s; see: docker compose logs vllm"; exit 1

serve-down:
	docker compose down

download:  ## Spider questions + SQLite DBs -> data/spider/
	uv run sqltune download

eval:  ## score a model on Spider: make eval CONFIG=experiments/configs/m0/baseline.yaml
	uv run sqltune eval --config $(CONFIG)

train:  ## fine-tune: make train CONFIG=experiments/configs/m0/qlora.yaml (GPU must be free of vLLM)
	@if curl -fsS localhost:8000/health >/dev/null 2>&1; then echo "vLLM is running and holds the GPU; run make serve-down first"; exit 1; fi
	uv run --extra train sqltune train --config $(CONFIG)

M1_TRAIN ?= lc_f0125 lc_f025 lc_f05 lc_f1 others ep3 lora16 seed2 seed3

m1-train:  ## M1a training sweep, sequential; a failed run is logged and skipped (~3 h GPU)
	@mkdir -p outputs
	@for r in $(M1_TRAIN); do \
	  echo "=== $$r $$(date +%H:%M)"; \
	  $(MAKE) --no-print-directory train CONFIG=experiments/configs/m1/train/$$r.yaml > outputs/$$r.log 2>&1 \
	    && echo "    ok $$(date +%H:%M)" || echo "    FAILED (see outputs/$$r.log)"; \
	done
