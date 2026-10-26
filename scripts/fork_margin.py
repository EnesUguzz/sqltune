"""How sure is a model when it gets a question wrong? Gold SQL vs the model's own SQL, scored
by teacher forcing on the training-time model (4-bit base + LoRA).

Purpose: The fine-tuned 7B misses ~28% of BIRD questions it was trained on (004). If, at the
first token where its answer and the gold answer part ways, it gives the gold token nearly
as much probability as its own, the miss is a near tie at an ambiguous fork (which column,
which join, x100 or not); more training would only memorize it. If the gold token gets
almost no probability, the model does not know the answer.
Flow: an eval run's predictions (failed questions; or correct ones as a control) -> the
same prompt as in training (build_messages, zero-shot) -> for gold and for the prediction:
fenced completion -> token log-probs -> first diverging token: p(gold token) vs p(own
token) under the same prefix, plus whole-answer log-prob per token -> summary on stdout.
Base and fine-tune are scored on the same questions.
Params: --train-config (base model, split), --adapter, --run (eval run under results/),
--correct (score correctly answered questions instead, as a control).
Try: `uv run python scripts/fork_margin.py --run m1_m1b_mix7b_fp8__zero__birdtrain`
(GPU, vLLM stopped).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-config", default="experiments/configs/m1b/train/mix7b.yaml")
    parser.add_argument("--adapter", default="adapters/m1b_mix7b")
    parser.add_argument("--run", required=True)
    parser.add_argument("--correct", action="store_true")
    args = parser.parse_args()

    import unsloth  # noqa: F401, I001  # must be imported before transformers / peft
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from sqltune.config.loader import load_config
    from sqltune.data.bird_train import load_bird_train
    from sqltune.prompting.builder import build_messages

    cfg = load_config(args.train_config)
    rows = [
        json.loads(x)
        for x in Path(f"results/{args.run}/predictions.jsonl").read_text().splitlines()
    ]
    rows = [
        r
        for r in rows
        if r["outcome"] != "gold_error" and bool(r["col_tolerant_correct"]) == args.correct
    ]
    by_id = {e.id: e for e in load_bird_train(cfg.data)}
    prompt_cfg = cfg.prompt.model_copy(update={"sample_rows": 0})

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.training.base_model,
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    def ids_of(messages: list[dict[str, str]], gen: bool = False) -> list[int]:
        out: Any = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def token_logprobs(m: Any, prompt: list[dict[str, str]], sql: str) -> tuple[list[int], Any]:
        """Completion token ids and their log-prob distributions (k x vocab)."""
        p = ids_of(prompt, gen=True)
        full = ids_of([*prompt, {"role": "assistant", "content": f"```sql\n{sql.strip()}\n```"}])
        assert full[: len(p)] == p
        k = len(full) - len(p)
        with torch.no_grad():
            out = m(input_ids=torch.tensor([full], device="cuda"), logits_to_keep=k + 1)
        return full[len(p) :], torch.log_softmax(out.logits[0, :-1].float(), dim=-1)

    def score(m: Any) -> list[dict[str, float]]:
        res = []
        for r in rows:
            ex = by_id[r["id"]]
            prompt = build_messages(ex, prompt_cfg)
            g_ids, g_lp = token_logprobs(m, prompt, ex.gold_sql)
            p_ids, p_lp = token_logprobs(m, prompt, r["pred_sql"] or "")
            g_tok = [g_lp[i, t].item() for i, t in enumerate(g_ids)]
            p_tok = [p_lp[i, t].item() for i, t in enumerate(p_ids)]
            fork = next(
                (i for i, (a, b) in enumerate(zip(g_ids, p_ids, strict=False)) if a != b),
                min(len(g_ids), len(p_ids)),
            )
            out = {
                "gold_lp_per_tok": sum(g_tok) / len(g_tok),
                "pred_lp_per_tok": sum(p_tok) / len(p_tok),
                "gold_lp_total": sum(g_tok),
                "pred_lp_total": sum(p_tok),
                "fork_pos": fork,
            }
            if fork < min(len(g_ids), len(p_ids)):  # same prefix: same distribution
                out["p_gold_at_fork"] = g_lp[fork, g_ids[fork]].exp().item()
                out["p_pred_at_fork"] = g_lp[fork, p_ids[fork]].exp().item()
                out["fork_gold_tok"] = tok.decode([g_ids[fork]])
                out["fork_pred_tok"] = tok.decode([p_ids[fork]])
            res.append(out)
        return res

    def report(name: str, res: list[dict[str, Any]]) -> None:
        forks = [x for x in res if "p_gold_at_fork" in x]
        pg = [x["p_gold_at_fork"] for x in forks]
        ratio = [x["p_gold_at_fork"] / max(x["p_pred_at_fork"], 1e-12) for x in forks]
        print(f"\n== {name}: {len(res)} questions, {len(forks)} with a diverging token")
        print(f"  p(gold token) at first fork: median {statistics.median(pg):.3f}")
        for lo, hi, label in (
            (0.2, 1.01, ">= 0.2"),
            (0.05, 0.2, "0.05-0.2"),
            (0.0, 0.05, "< 0.05"),
        ):
            print(f"    {label:9} {sum(lo <= v < hi for v in pg)}")
        print(
            f"  gold within 3x of own choice at fork: {sum(r >= 1 / 3 for r in ratio)} / {len(forks)}"
        )
        print(
            f"  whole answer, mean log-prob per token: gold {statistics.mean(x['gold_lp_per_tok'] for x in res):.3f}, "
            f"own {statistics.mean(x['pred_lp_per_tok'] for x in res):.3f}"
        )
        print(
            f"  gold more likely than own (total log-prob): {sum(x['gold_lp_total'] > x['pred_lp_total'] for x in res)} / {len(res)}"
        )

    print(f"{args.run}: {'correct' if args.correct else 'failed'} questions, n={len(rows)}")
    base = score(model)
    report("base (no adapter)", base)
    model = PeftModel.from_pretrained(model, args.adapter)
    tuned = score(model)
    report(f"fine-tune ({args.adapter})", tuned)
    print("\n  fine-tune, first forks (gold token | own token | p_gold | p_own):")
    for r, x in list(zip(rows, tuned, strict=True))[:25]:
        if "p_gold_at_fork" in x:
            print(
                f"    {r['id']}  {x['fork_gold_tok']!r:14} | {x['fork_pred_tok']!r:14} | "
                f"{x['p_gold_at_fork']:.3f} | {x['p_pred_at_fork']:.3f}"
            )


if __name__ == "__main__":
    main()
