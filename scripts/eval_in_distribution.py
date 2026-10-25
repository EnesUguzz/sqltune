"""Execution accuracy on BIRD *train*-distribution questions: the held-out validation databases
of the M1b split, or a sample of the training questions themselves.

Purpose: Separates "the fine-tune learned nothing" from "it learned, but the gain does not
transfer to Mini-Dev". If base and fine-tune tie even on questions the adapter trained on,
the model is undertrained; if the fine-tune wins on train / validation but not on Mini-Dev,
the gain is distribution-specific. Not a test set: these questions never decide anything
else.
Flow: train config (data split, seed) -> load_bird_train -> select_fraction -> split_by_db
-> val databases, or a seeded sample of the train part -> run_eval with an existing eval
config's model and prompt -> results/<run-name>/.
The saved results/<run>/config.yaml is the borrowed eval config (it says bird_split diag);
the questions actually used come from this script.
Params: --train-config (the split, typical m1b/train/mix7b.yaml), --eval-config (model and
prompt variant), --set (val | train), --n (train sample size, typical 250), --run-name.
Try: `uv run python scripts/eval_in_distribution.py --eval-config
experiments/configs/m1b/fp8check/zero__birddiag.yaml --set val --run-name m1_m1b_mix7b_fp8__zero__birdval`
(with the matching model served).
"""

from __future__ import annotations

import argparse
import random

from sqltune import runner
from sqltune.config.loader import load_config
from sqltune.data.bird_train import load_bird_train
from sqltune.data.subsets import select_fraction, split_by_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-config", default="experiments/configs/m1b/train/mix7b.yaml")
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--set", choices=["val", "train"], required=True)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    t = load_config(args.train_config)
    cfg = load_config(args.eval_config)
    cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"run_name": args.run_name})})
    examples = select_fraction(load_bird_train(t.data), t.training.train_fraction, t.training.seed)
    train, val = split_by_db(examples, t.training.val_fraction, t.training.seed)
    chosen = val if args.set == "val" else random.Random(0).sample(train, args.n)
    print(f"{args.set}: {len(chosen)} questions on {len({e.db_id for e in chosen})} databases")
    runner.load_examples = lambda _cfg: chosen  # type: ignore[assignment]
    print(runner.run_eval(cfg))


if __name__ == "__main__":
    main()
