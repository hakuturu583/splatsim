"""Spawn-only scenario: spawn ego vehicle at a lanelet coordinate and wait.

This is the simplest possible scenario — it spawns the ego vehicle at
the configured Lanelet2 pose, attaches the spectator camera, and waits
until the timeout expires.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=100 ego.spawn_s=5.0
"""

from __future__ import annotations

import logging

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)

logger = logging.getLogger(__name__)


class SpawnOnlyScenario(BaseScenario):
    """Spawn the ego at a Lanelet2 pose and hold until timeout.

    The scenario:

    1. Converts the Lanelet2 pose to a CARLA world pose via OpenDRIVE.
    2. Spawns the ego vehicle at the snapped position.
    3. Attaches the spectator camera to follow the ego.
    4. Waits for ``timeout_seconds`` then passes.
    """

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        *,
        timeout_seconds: float = 30.0,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
        )
        self._timeout_seconds = timeout_seconds

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
            self._timeout_seconds,
        )

        self.register_pass_condition(
            TimeoutCondition(self._timeout_seconds, label="spawn_hold_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by the timeout condition."""
        return False
