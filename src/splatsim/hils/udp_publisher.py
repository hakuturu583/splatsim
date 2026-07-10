"""UDP transport for Hesai HILS point-cloud packets.

Wraps :mod:`splatsim.hils.hesai_packet` with a UDP socket so a rendered
range image can be streamed to a real LiDAR driver as if it came from the
physical sensor. Pure standard-library networking — no DDS/CARLA/torch
dependency — so it works in the core (headless) install.
"""

from __future__ import annotations

import socket
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from splatsim.hils.hesai_packet import (
    RETURN_MODE_STRONGEST,
    HesaiModel,
    build_packets,
    get_model,
)


def _utc_date_time(epoch_s: float) -> tuple[tuple[int, int, int, int, int, int], int]:
    """Split a Unix timestamp into a Hesai UTC date-time tuple + microseconds.

    Returns ``((year-1900, month, day, hour, minute, second), microseconds)``
    where ``microseconds`` is the sub-second part (0..999_999), matching the
    physical sensor's tail where the timestamp field counts microseconds
    within the second described by the date-time bytes.
    """
    whole = int(epoch_s)
    micros = int(round((epoch_s - whole) * 1_000_000)) % 1_000_000
    tm = time.gmtime(whole)
    date_time = (
        tm.tm_year - 1900,
        tm.tm_mon,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
    )
    return date_time, micros


class HesaiHilsPublisher:
    """Streams rendered LiDAR range images as Hesai UDP data packets.

    Parameters
    ----------
    sensor_type:
        ``"OT128"`` or ``"XT32"`` — selects the wire format.
    host:
        Destination IP (unicast or broadcast) for the UDP packets.
    port:
        Destination UDP port (physical Hesai default is ``2368``).
    start_epoch_s:
        Wall-clock (Unix) time that simulation time ``0`` maps to. Packet
        timestamps are ``start_epoch_s + sim_time_s`` so the wire clock is
        the simulation's internal clock. Defaults to ``time.time()`` at
        construction (i.e. the sim starts "now"); pass an explicit value to
        pin the sensor's date-time to a fixed epoch.
    return_mode:
        Return-mode byte written into every packet tail.
    """

    def __init__(
        self,
        sensor_type: str,
        *,
        host: str = "127.0.0.1",
        port: int = 2368,
        start_epoch_s: Optional[float] = None,
        return_mode: int = RETURN_MODE_STRONGEST,
    ) -> None:
        self._model: HesaiModel = get_model(sensor_type)
        self._addr = (host, int(port))
        self._start_epoch_s = (
            time.time() if start_epoch_s is None else float(start_epoch_s)
        )
        self._return_mode = return_mode
        self._sequence = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow broadcast destinations (e.g. 255.255.255.255) out of the box.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    @property
    def model(self) -> HesaiModel:
        return self._model

    @property
    def address(self) -> tuple[str, int]:
        return self._addr

    @property
    def start_epoch_s(self) -> float:
        return self._start_epoch_s

    def build(
        self,
        *,
        distance_m: NDArray[np.floating],
        intensity: NDArray[np.floating],
        valid: NDArray[np.bool_],
        azimuth_rad: NDArray[np.floating],
        spin_hz: float,
        sim_time_s: float = 0.0,
    ) -> list[bytes]:
        """Encode one range-image frame into packets (without sending).

        ``sim_time_s`` is the simulation-internal elapsed time (seconds); the
        packet date-time is ``start_epoch_s + sim_time_s``.
        """
        date_time, micros = _utc_date_time(self._start_epoch_s + sim_time_s)
        packets = build_packets(
            self._model,
            distance_m=distance_m,
            intensity=intensity,
            valid=valid,
            azimuth_rad=azimuth_rad,
            motor_speed_rpm=int(round(spin_hz * 60.0)),
            timestamp_us=micros,
            date_time=date_time,
            return_mode=self._return_mode,
            seq_start=self._sequence,
        )
        self._sequence = (self._sequence + len(packets)) & 0xFFFFFFFF
        return packets

    def publish(
        self,
        *,
        distance_m: NDArray[np.floating],
        intensity: NDArray[np.floating],
        valid: NDArray[np.bool_],
        azimuth_rad: NDArray[np.floating],
        spin_hz: float,
        sim_time_s: float = 0.0,
    ) -> int:
        """Encode a range-image frame and send every packet over UDP.

        Args mirror :meth:`build`. Returns the number of packets sent.
        """
        packets = self.build(
            distance_m=distance_m,
            intensity=intensity,
            valid=valid,
            azimuth_rad=azimuth_rad,
            spin_hz=spin_hz,
            sim_time_s=sim_time_s,
        )
        for pkt in packets:
            self._sock.sendto(pkt, self._addr)
        return len(packets)

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> HesaiHilsPublisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["HesaiHilsPublisher"]
