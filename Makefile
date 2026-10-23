CONFIG ?= experiments/configs/base.yaml
MODEL ?= Qwen/Qwen2.5-Coder-3B-Instruct
LORA ?=
# Seconds to wait for vLLM to report healthy (first start downloads the model).
SERVE_WAIT ?= 900

.PHONY: setup test lint format check-env serve serve-down

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
