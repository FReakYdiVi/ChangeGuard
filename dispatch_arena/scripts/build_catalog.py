"""Build the dispatch arena scenario catalog.

Usage:
    .venv/bin/python -m dispatch_arena.scripts.build_catalog
    .venv/bin/python -m dispatch_arena.scripts.build_catalog --backend llm
    .venv/bin/python -m dispatch_arena.scripts.build_catalog --easy 30 --medium 40 --hard 30 --out catalog.json

Backends:
  anchor (default): deterministic, no API key needed
  llm             : uses Claude (anthropic SDK, requires ANTHROPIC_API_KEY).
                    Falls back per-scenario to anchors if the LLM trips.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dispatch_arena.catalog.builder import (
    AnchorScenarioBuilder,
    LLMScenarioBuilder,
    save_catalog,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the Dispatch Arena scenario catalog.")
    parser.add_argument("--backend", choices=["anchor", "llm"], default="anchor")
    parser.add_argument("--easy", type=int, default=30)
    parser.add_argument("--medium", type=int, default=40)
    parser.add_argument("--hard", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed for determinism")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "catalog" / "catalog.json",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    counts = {"easy": args.easy, "medium": args.medium, "hard": args.hard}
    total = sum(counts.values())

    if args.backend == "llm":
        builder = LLMScenarioBuilder(master_seed=args.seed)
    else:
        builder = AnchorScenarioBuilder(master_seed=args.seed)

    print(f"Building {total} scenarios via {args.backend} backend ({counts}) -> {args.out}")
    specs = builder.build_batch(counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_catalog(specs, args.out)

    # Summary
    by_diff = {}
    by_skill = {}
    for s in specs:
        by_diff[s.difficulty] = by_diff.get(s.difficulty, 0) + 1
        for t in s.skill_focus:
            by_skill[t] = by_skill.get(t, 0) + 1
    print(f"\n=== CATALOG WRITTEN: {len(specs)} scenarios ===")
    print(f"By difficulty: {by_diff}")
    print(f"By skill_focus tag: {dict(sorted(by_skill.items(), key=lambda kv: -kv[1]))}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
