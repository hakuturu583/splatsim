"""UDP transport for Hesai HILS point-cloud packets.

Wraps :mod:`splatsim.hils.hesai_packet` with a UDP socket so a rendered
range image can be streamed to a real LiDAR driver as if it came from the
physical sensor. The range image is encoded with ``torch`` (on the GPU when
the tensors are CUDA); only the final packed byte buffer is moved to the host,
once, for the UDP send.
"""

from __future__ import annotations

import socket
import time
from typing import TYPE_CHECKING, Optional

import torch

from splatsim.hils.hesai_packet import (
    RETURN_MODE_STRONGEST,
    HesaiModel,
    build_frame_tensor,
    get_model,
)

if TYPE_CHECKING:
    import numpy as np


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

    def build_array(
        self,
        *,
        distance_m: torch.Tensor,
        intensity: torch.Tensor,
        valid: torch.Tensor,
        azimuth_rad: torch.Tensor,
        spin_hz: float,
        sim_time_s: float = 0.0,
    ) -> "np.ndarray":
        """Encode one frame into a host-side ``(n_packets, packet_size)`` grid.

        Inputs are ``torch`` tensors (CUDA -> encoded on the GPU). The single
        device->host copy happens here; each returned row is a ready-to-send
        packet. ``sim_time_s`` is the simulation-internal elapsed time
        (seconds); the packet date-time is ``start_epoch_s + sim_time_s``.
        Advances the UDP sequence counter.
        """
        date_time, micros = _utc_date_time(self._start_epoch_s + sim_time_s)
        buf = build_frame_tensor(
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
        self._sequence = (self._sequence + buf.shape[0]) & 0xFFFFFFFF
        return buf.cpu().numpy()

    def publish(
        self,
        *,
        distance_m: torch.Tensor,
        intensity: torch.Tensor,
        valid: torch.Tensor,
        azimuth_rad: torch.Tensor,
        spin_hz: float,
        sim_time_s: float = 0.0,
    ) -> int:
        """Encode a range-image frame and send every packet over UDP.

        Sends each packet straight from its row in the host-side byte grid (via
        the buffer protocol — no per-packet ``bytes`` copy). Returns the
        number of packets sent.
        """
        buf = self.build_array(
            distance_m=distance_m,
            intensity=intensity,
            valid=valid,
            azimuth_rad=azimuth_rad,
            spin_hz=spin_hz,
            sim_time_s=sim_time_s,
        )
        addr = self._addr
        sendto = self._sock.sendto
        for row in buf:
            sendto(row, addr)
        return buf.shape[0]

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> HesaiHilsPublisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["HesaiHilsPublisher"]
