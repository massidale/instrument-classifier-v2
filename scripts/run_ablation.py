#!/usr/bin/env python
"""Per-branch ablation study: train N configurations that differ ONLY in the
active branches (same seed, split, epochs, threshold procedure) and write a
comparison table to outputs/ablation.md.

    python scripts/run_ablation.py --config configs/default.yaml
    python scripts/run_ablation.py --config configs/default.yaml --epochs 2  # smoke run
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from instrument_classifier.train import run_training
from instrument_classifier.utils import load_config

ABLATION_COMBOS = [
    {"mel": True, "cqt": False, "wave": False, "chroma": False},
    {"mel": True, "cqt": True, "wave": False, "chroma": False},
    {"mel": True, "cqt": True, "wave": True, "chroma": False},
    {"mel": True, "cqt": True, "wave": True, "chroma": True},
]


def combo_name(combo: dict[str, bool]) -> str:
    order = ("mel", "cqt", "wave", "chroma")  # report order, not alphabetical
    return "+".join(k for k in order if combo[k])


def format_ablation_table(rows: list[dict]) -> str:
    lines = ["| Branches | val micro-F1 | test micro-F1 | test macro-F1 |",
             "|---|---|---|---|"]
    for r in rows:
        test = r.get("test") or {}
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "n/a"
        lines.append(f"| {r['name']} | {fmt(r['val_micro_f1'])} "
                     f"| {fmt(test.get('micro_f1'))} | {fmt(test.get('macro_f1'))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override finetune_epochs (smoke runs)")
    args = parser.parse_args()
    base = load_config(args.config)

    rows = []
    for combo in ABLATION_COMBOS:
        name = combo_name(combo)
        cfg = copy.deepcopy(base)          # identical except for what follows
        cfg["branches"] = dict(combo)
        cfg["output_dir"] = str(Path(base["output_dir"]) / "ablation" / name)
        if args.epochs is not None:
            cfg["train"]["finetune_epochs"] = args.epochs
        print(f"=== ablation: {name} ===")
        results = run_training(cfg)
        rows.append({"name": name, "val_micro_f1": results["val_micro_f1"],
                     "test": results.get("test")})

    out = Path(base["output_dir"]) / "ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Ablation study — branch contributions\n\n"
                   + format_ablation_table(rows))
    print(f"Wrote {out}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
