"""Entry point for the odaibatest spawn-only scenario.

Registers :class:`SpawnOnlyScenario` via the upstream scenario registry
and delegates all orchestration to
:func:`autoware_carla_scenario.examples.run.run_scenario`.

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

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario.examples.run import register_scenario, run_scenario

from .spawn_only import SpawnOnlyConfig, SpawnOnlyScenario

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=True)

register_scenario("spawn_only", SpawnOnlyScenario, SpawnOnlyConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point — delegates to the upstream run_scenario."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    result = run_scenario(cfg)
    sys.exit(0 if result.passed else 1)


def main() -> None:
    """CLI entry point."""
    _hydra_main()


if __name__ == "__main__":
    main()
