"""The ROS 2 domain splatsim's DDS entities join.

splatsim reaches Autoware with CycloneDDS directly rather than through rclpy, so
nothing reads ``ROS_DOMAIN_ID`` on its behalf: a participant built with no domain
takes CycloneDDS's default, domain 0. A stack running on any other domain -- a
simulator sandbox that gives each run its own, say -- then never hears the point
clouds, and both ends look healthy while it happens: splatsim reports renders and
publishes, the subscriber reports no publisher.

So build participants here instead, honouring the variable every other ROS 2
process honours.
"""

from __future__ import annotations

import os
import re
from typing import Mapping

from cyclonedds.domain import DomainParticipant

#: What a ROS 2 process uses when ``ROS_DOMAIN_ID`` is unset -- and what
#: CycloneDDS uses when nothing names a domain, so the default is unchanged.
DEFAULT_DOMAIN_ID = 0
#: The largest domain id DDS allows (ROS 2 recommends staying below 102).
MAX_DOMAIN_ID = 232

ENV_VAR = "ROS_DOMAIN_ID"

# An environment value is not a Python literal, so it is not read like one:
# ``int`` would take "1_0" for 10 and the full-width "１０" for 10 as well. A
# domain id is decimal digits and nothing else.
_DECIMAL = re.compile(r"[0-9]+")


def ros_domain_id(environ: Mapping[str, str] | None = None) -> int:
    """The domain ``ROS_DOMAIN_ID`` names, or :data:`DEFAULT_DOMAIN_ID`.

    Unset or empty means the default, as it does for rclpy. A value that is not
    a domain raises rather than falling back: publishing on the wrong domain is
    silent, and a typo that costs an hour of debugging is worse than a refusal
    to start.
    """
    raw = (environ if environ is not None else os.environ).get(ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_DOMAIN_ID
    if not _DECIMAL.fullmatch(raw):
        raise ValueError(
            f"{ENV_VAR}={raw!r} is not a domain id: expected decimal digits "
            f"0..{MAX_DOMAIN_ID}"
        )
    domain_id = int(raw)
    if domain_id > MAX_DOMAIN_ID:
        raise ValueError(f"{ENV_VAR}={domain_id} is outside 0..{MAX_DOMAIN_ID}")
    return domain_id


def make_participant() -> DomainParticipant:
    """A ``DomainParticipant`` on the domain ``ROS_DOMAIN_ID`` names."""
    return DomainParticipant(ros_domain_id())
