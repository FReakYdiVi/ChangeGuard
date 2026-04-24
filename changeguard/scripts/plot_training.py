"""Plot GRPO training curves from train_grpo.py log files.

Parses the per-step metric dicts TRL prints to stdout and renders a 2x3 grid
of training diagnostics: reward, reward_std, loss, grad_norm, tools failure,
and completion length. Supports comparing multiple runs on the same axes.

Usage:
    python -m scripts.plot_training <label>=<logfile> [<label>=<logfile> ...] [--out png_path]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

METRIC_DICT_RE = re.compile(r"\{'loss':.*?\}")


def parse_log(path: Path) -> List[Dict[str, float]]:
    text = path.read_text()
    rows: List[Dict[str, float]] = []
    for match in METRIC_DICT_RE.finditer(text):
        try:
            d = ast.literal_eval(match.group(0))
        except Exception:
            continue
        numeric = {}
        for k, v in d.items():
            try:
                numeric[k] = float(v)
            except (TypeError, ValueError):
                pass
        rows.append(numeric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="label=path specs")
    parser.add_argument("--out", default="artifacts/training_curves.png")
    args = parser.parse_args()

    runs: Dict[str, List[Dict[str, float]]] = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        runs[label] = parse_log(Path(path))
        if not runs[label]:
            print(f"warning: no metric rows found in {path}", file=sys.stderr)

    panels = [
        ("reward (group mean)", "rewards/env_reward/mean", "reward"),
        ("reward std (exploration signal)", "rewards/env_reward/std", "reward_std"),
        ("loss", "loss", None),
        ("grad_norm", "grad_norm", None),
        ("tools/failure_frequency (lower better)", "tools/failure_frequency", None),
        ("completions/mean_length", "completions/mean_length", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, primary_key, fallback_key) in zip(axes.flat, panels):
        for label, rows in runs.items():
            xs = list(range(1, len(rows) + 1))
            ys = []
            for r in rows:
                v = r.get(primary_key)
                if v is None and fallback_key is not None:
                    v = r.get(fallback_key)
                ys.append(v if v is not None else float("nan"))
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if ax is axes.flat[0]:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("ChangeGuard GRPO training optimization", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
