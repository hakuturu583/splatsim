from __future__ import annotations

import pytest

pytest.importorskip("cyclonedds")

from splatsim.cyclonedds.domain import (  # noqa: E402
    DEFAULT_DOMAIN_ID,
    ENV_VAR,
    MAX_DOMAIN_ID,
    ros_domain_id,
)


def test_unset_is_the_ros_default() -> None:
    assert ros_domain_id({}) == DEFAULT_DOMAIN_ID


def test_empty_is_unset() -> None:
    assert ros_domain_id({ENV_VAR: "   "}) == DEFAULT_DOMAIN_ID


def test_a_domain_is_read() -> None:
    assert ros_domain_id({ENV_VAR: "62"}) == 62


def test_surrounding_whitespace_is_ignored() -> None:
    assert ros_domain_id({ENV_VAR: " 7\n"}) == 7


def test_the_highest_domain_is_allowed() -> None:
    assert ros_domain_id({ENV_VAR: str(MAX_DOMAIN_ID)}) == MAX_DOMAIN_ID


@pytest.mark.parametrize("value", ["twelve", "1.5", "0x2a"])
def test_a_value_that_is_not_a_domain_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match=ENV_VAR):
        ros_domain_id({ENV_VAR: value})


@pytest.mark.parametrize("value", ["-1", str(MAX_DOMAIN_ID + 1)])
def test_a_domain_out_of_range_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="outside"):
        ros_domain_id({ENV_VAR: value})


def test_the_process_environment_is_the_default_source(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "42")
    assert ros_domain_id() == 42
