"""Internal runtime state containers for ChangeGuard.

These structures intentionally include hidden state that must not be returned in
OpenEnv observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .models import EpisodeConfig, EpisodeStats, RolloutStage, TenantHiddenState, TenantId, TenantVisibleState


@dataclass
class TenantRuntimeState:
    """Internal tenant state pairing visible and hidden components.

    TODO: Wire transitions so hidden/visible pieces evolve consistently.
    """

    visible: TenantVisibleState
    hidden: TenantHiddenState


@dataclass
class EpisodeRuntimeState:
    """Full mutable episode state for native environment engine.

    TODO: Use this as the sole mutable state object in server environment logic.
    """

    config: EpisodeConfig
    stage: RolloutStage = RolloutStage.PLAN
    tenants: Dict[TenantId, TenantRuntimeState] = field(default_factory=dict)
    stats: EpisodeStats = field(default_factory=EpisodeStats)
    approval_granted_c: bool = False
