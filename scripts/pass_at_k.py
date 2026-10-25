"""Sample k answers per question: pass@k (headroom for RL / reranking) and execution-based
majority voting (self-consistency), compared with the greedy run of the same model.

Purpose: Greedy EX is one draw. If one of k sampled answers is often right (pass@k well
above greedy), training on execution feedback (GRPO) or reranking has room to help; if not,
the model does not have the answer at all. Majority voting over execution results is a
no-training way to use that room.
Flow: eval config (model, prompt, eval set) or the BIRD validation split -> k identical
requests per question at --temperature -> extract SQL -> execute every sample (process
pool) -> per question: correct samples, pass@k, vote = answer of the largest group of
samples with the same result set (ties: first seen) -> summary + McNemar of vote vs the
greedy run -> results/<run-name>/passk.json.
Params: --eval-config, --set (config | birdval), --k (typical 8), --temperature (typical
0.7), --greedy-run (results dir of the greedy run to compare with), --run-name.
Test sets are refused (same guard as every eval).
Try: `uv run python scripts/pass_at_k.py --eval-config
experiments/configs/m1/eval/m1b--m1b_mix__zero__birddiag.yaml --greedy-run
m1_m1b_mix__zero__birddiag --run-name m1_m1b_mix__zero__birddiag_k8` (model served).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sqltune import runner
from sqltune.config.loader import load_config
from sqltune.config.models import EvalConfig
from sqltune.data.bird_train import load_bird_train
from sqltune.data.subsets import select_fraction, split_by_db
from sqltune.domain import Example
from sqltune.evaluation.executor import score_example
from sqltune.evaluation.metrics import mcnemar_exact_p
from sqltune.inference.extract import extract_sql
from sqltune.inference.openai_compat import OpenAICompatBackend


def _score(job: tuple[Example, str | None, EvalConfig]) -> tuple[bool, str | None]:
    """(column-order-tolerant correct, result-set key for voting or None if it did not run)."""
    ex, sql, cfg = job
    _, tolerant, pred = score_example(ex, sql, cfg)
    if pred is None or not pred.ok or pred.rows is None:
        return tolerant, None
    rows = sorted({repr(sorted(map(repr, r))) for r in pred.rows})  # set, column order ignored
    return tolerant, json.dumps(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--set", choices=["config", "birdval"], default="config")
    parser.add_argument("--train-config", default="experiments/configs/m1b/train/mix7b.yaml")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--greedy-run", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    cfg = load_config(args.eval_config)
    cfg = cfg.model_copy(
        update={"inference": cfg.inference.model_copy(update={"temperature": args.temperature})}
    )
    if args.set == "birdval":
        t = load_config(args.train_config)
        examples = select_fraction(
            load_bird_train(t.data), t.training.train_fraction, t.training.seed
        )
        examples = split_by_db(examples, t.training.val_fraction, t.training.seed)[1]
    else:
        examples = runner.load_examples(cfg)  # refuses test sets
    requests = runner.build_requests(examples, cfg)
    flat = [m for m in requests for _ in range(args.k)]
    comps = OpenAICompatBackend(cfg.inference).complete_many(flat)
    n_err = sum(c.error is not None for c in comps)
    sqls = [extract_sql(c.text) for c in comps]
    jobs = [(examples[i // args.k], s, cfg.evaluation) for i, s in enumerate(sqls)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=cfg.evaluation.exec_workers, mp_context=ctx) as pool:
        scored = list(pool.map(_score, jobs, chunksize=16))

    greedy_path = Path("results") / args.greedy_run / "predictions.jsonl"
    greedy = {r["id"]: r for r in map(json.loads, greedy_path.read_text().splitlines())}
    per_q: list[dict[str, Any]] = []
    for qi, ex in enumerate(examples):
        samples = scored[qi * args.k : (qi + 1) * args.k]
        if greedy[ex.id]["outcome"] == "gold_error":
            continue
        groups = Counter(key for _, key in samples if key is not None)
        if groups:
            top = groups.most_common(1)[0][0]
            vote = next(ok for ok, key in samples if key == top)
        else:
            vote = samples[0][0]
        per_q.append(
            {
                "id": ex.id,
                "n_correct": sum(ok for ok, _ in samples),
                "vote": bool(vote),
                "greedy": bool(greedy[ex.id]["col_tolerant_correct"]),
            }
        )
    n = len(per_q)
    fixed = sum(q["vote"] and not q["greedy"] for q in per_q)
    broken = sum(q["greedy"] and not q["vote"] for q in per_q)
    summary = {
        "n": n,
        "k": args.k,
        "temperature": args.temperature,
        "transport_errors": n_err,
        "greedy_ex": 100 * sum(q["greedy"] for q in per_q) / n,
        "sampled_ex": 100 * sum(q["n_correct"] for q in per_q) / (n * args.k),
        "pass_at_k": 100 * sum(q["n_correct"] > 0 for q in per_q) / n,
        "vote_ex": 100 * sum(q["vote"] for q in per_q) / n,
        "vote_vs_greedy": {"fixed": fixed, "broken": broken, "p": mcnemar_exact_p(fixed, broken)},
        "never_correct": sum(q["n_correct"] == 0 for q in per_q),
        "always_correct": sum(q["n_correct"] == args.k for q in per_q),
    }
    out = Path("results") / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "passk.json").write_text(json.dumps({"summary": summary, "questions": per_q}, indent=1))
    print(args.run_name, json.dumps(summary))


if __name__ == "__main__":
    main()
