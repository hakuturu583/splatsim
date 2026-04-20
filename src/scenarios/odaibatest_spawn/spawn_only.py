"""Spawn-only scenario: spawn ego vehicle at a lanelet coordinate and wait.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=2223437 ego.spawn_s=5.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)

logger = logging.getLogger(__name__)


@dataclass
class SpawnOnlyConfig:
    """Parameters for the spawn-only scenario."""

    name: str = "spawn_only"
    timeout_seconds: float = 30.0


class SpawnOnlyScenario(BaseScenario):
    """Spawn the ego at a Lanelet2 pose and hold until timeout."""

    def __init__(
        self,
        ego_config: EgoConfig,
        *,
        config: SpawnOnlyConfig,
        spawn_pose: Lanelet2Pose,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
        )
        self._config = config

    def setup(self) -> None:
        """Snap ego spawn to CARLA road and attach spectator."""
        self._setup_ego_spawn()

        ego_actor_name = EGO_ROLE_NAME
        self.follow_with_spectator(
            lambda: self.world.get_actors().filter(f"*{ego_actor_name}*")[0]
            if self.world.get_actors().filter(f"*{ego_actor_name}*")
            else None,
        )

        assert self._spawn_pose is not None  # noqa: S101
        logger.info(
            "Ego spawned on lanelet %d (s=%.1f). Waiting %.1f s ...",
            self._spawn_pose.lanelet_id,
            self._spawn_pose.s,
            self._config.timeout_seconds,
        )

        self.register_pass_condition(
            TimeoutCondition(self._config.timeout_seconds, label="spawn_hold_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by the timeout condition."""
        return False
