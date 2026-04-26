"""Catalog -> HuggingFace Dataset wiring for GRPO training.

Two important contracts maintained here:

1. **No metadata leak into the prompt**: the agent's prompt only carries
   the system instructions + a "Begin shift" user turn. Difficulty,
   skill_focus, and theme live in `_`-prefixed columns the trainer can
   read for stratified eval / per-bucket logging, but they NEVER end up
   templated into the model's context. (This matches what we agreed on
   in the catalog discussion: tags are dataset metadata, not features.)

2. **Stratified split by difficulty** so the eval set covers every
   difficulty bucket proportionally. Same `master_seed` -> same split,
   bit-for-bit, which is required for "trained vs. baseline on same
   eval scenarios" to be a defensible claim.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset

from dispatch_arena.catalog.builder import load_catalog
from dispatch_arena.catalog.spec import ScenarioSpec


def stratified_split(
    specs: List[ScenarioSpec],
    eval_fraction: float = 0.30,
    master_seed: int = 0,
) -> Tuple[List[ScenarioSpec], List[ScenarioSpec]]:
    """Split scenarios into train/eval, keeping difficulty proportions stable.

    A `master_seed` fully determines the split — re-running with the same
    seed yields the same train/eval partition.
    """
    rng = random.Random(master_seed)
    by_difficulty: dict = {}
    for spec in specs:
        by_difficulty.setdefault(spec.difficulty, []).append(spec)

    train: List[ScenarioSpec] = []
    eval_: List[ScenarioSpec] = []
    for difficulty, group in by_difficulty.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_eval = max(1, int(round(len(shuffled) * eval_fraction)))
        eval_.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def build_hf_dataset(
    specs: List[ScenarioSpec],
    system_prompt: str,
    user_kickoff: str = "Begin the shift. Use the dashboard tool to get the current state, then dispatch.",
) -> Dataset:
    """Convert a list of scenarios into a HuggingFace Dataset.

    Each row carries:
      - `prompt`         : list[chat-message] passed to the model. Only
                           system + user kickoff. NO difficulty/theme leak.
      - `seed`           : the env reset seed
      - `_difficulty`    : "easy" | "medium" | "hard"   [eval bucketing]
      - `_skill_focus`   : list[str]                     [eval bucketing]
      - `_name`          : scenario name                 [debugging]
      - `_config`        : dict of Config kwargs         [env reset]
    """
    rows = []
    for spec in specs:
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_kickoff},
                ],
                "seed": spec.seed,
                "_difficulty": spec.difficulty,
                "_skill_focus": list(spec.skill_focus),
                "_name": spec.name,
                "_config": {
                    "mode": "normal",
                    "max_ticks": spec.max_ticks,
                    "num_couriers": spec.num_couriers,
                    "num_orders": spec.num_orders,
                    "scenario_bucket": spec.scenario_bucket,
                    "rolling_arrivals": spec.rolling_arrivals,
                    "traffic_noise": spec.traffic_noise,
                    "visible_prep": spec.visible_prep,
                },
            }
        )
    return Dataset.from_list(rows)


def load_catalog_datasets(
    catalog_path: Path,
    system_prompt: str,
    eval_fraction: float = 0.30,
    master_seed: int = 0,
) -> Tuple[Dataset, Dataset, List[ScenarioSpec], List[ScenarioSpec]]:
    """Top-level helper: load catalog.json -> (train_ds, eval_ds, train_specs, eval_specs)."""
    specs = load_catalog(catalog_path)
    train_specs, eval_specs = stratified_split(specs, eval_fraction, master_seed)
    train_ds = build_hf_dataset(train_specs, system_prompt)
    eval_ds = build_hf_dataset(eval_specs, system_prompt)
    return train_ds, eval_ds, train_specs, eval_specs
