"""Entry point for the odaibatest spawn-only scenario.

Reuses :func:`autoware_carla_scenario.examples.run.run_scenario` for the
ScenarioQueue orchestration.  Only ``build_scenario`` is overridden to
instantiate :class:`SpawnOnlyScenario`.

Usage::

    source .env
    uv run spawn-scenario
    uv run spawn-scenario ego.spawn_lanelet_id=200 ego.spawn_s=10.0
    uv run spawn-scenario scenario.timeout_seconds=60.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import carla
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario import (
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    ScenarioQueue,
    SpawnTransform,
)

from .spawn_only import SpawnOnlyScenario

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def build_scenario(cfg: DictConfig) -> tuple[EgoConfig, SpawnOnlyScenario]:
    """Build ego config, spawn pose, and scenario from a resolved Hydra config.

    The EgoConfig / Lanelet2Pose assembly mirrors
    :func:`autoware_carla_scenario.examples.run.build_scenario` — the only
    difference is the scenario class instantiated at the end.
    """
    ground_projection = GroundProjectionConfig(
        ray_distance_upper=float(cfg.entity.ground_projection_ray_distance_upper),
        ray_distance_lower=float(cfg.entity.ground_projection_ray_distance_lower),
    )

    ego = EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type=cfg.ego.vehicle_type,
        initial_speed_kmh=float(cfg.ego.initial_speed_kmh),
        spawn_retry_max_count=int(cfg.entity.spawn_retry_max_count),
        spawn_retry_t_step=float(cfg.entity.spawn_retry_t_step),
        spawn_retry_z_step=float(cfg.entity.spawn_retry_z_step),
    )

    spawn_pose = Lanelet2Pose(
        lanelet_id=cfg.ego.spawn_lanelet_id,
        s=cfg.ego.spawn_s,
    )

    return ego, SpawnOnlyScenario(
        ego,
        spawn_pose,
        timeout_seconds=float(cfg.scenario.timeout_seconds),
        ground_projection=ground_projection,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point.

    ScenarioQueue orchestration follows the same pattern as
    :func:`autoware_carla_scenario.examples.run.run_scenario`.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    _ego, scenario = build_scenario(cfg)

    xodr_path = Path(cfg.map.xodr_path) if cfg.map.get("xodr_path") else None
    lanelet2_path = (
        Path(cfg.map.lanelet2_path) if cfg.map.get("lanelet2_path") else None
    )

    queue = ScenarioQueue(
        host=cfg.server.host,
        port=cfg.server.port,
        tm_port=cfg.traffic_manager.port,
        xodr_path=xodr_path,
        lanelet2_path=lanelet2_path,
        map_name=cfg.map.name,
        cooldown_seconds=float(cfg.server.get("cooldown_seconds", 0.0)),
        cooldown_max_retries=int(cfg.server.get("cooldown_max_retries", 0)),
    )
    queue.add(scenario)

    with queue:
        results = queue.run_all()

    result = results[0]
    status = "PASSED" if result.passed else "FAILED"
    print(f"{status}: {result.message} ({result.elapsed_seconds:.2f}s)")  # noqa: T201
    sys.exit(0 if result.passed else 1)


def main() -> None:
    """CLI entry point."""
    _hydra_main()


if __name__ == "__main__":
    main()
