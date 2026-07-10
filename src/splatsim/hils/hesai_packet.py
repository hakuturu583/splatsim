"""Hesai LiDAR UDP point-cloud packet encoders for HILS.

Hardware-in-the-loop simulation feeds a real LiDAR driver (e.g. Autoware
``nebula``, ``HesaiLidar_ROS``) the exact UDP data packets that the physical
sensor would emit on the wire, instead of a decoded ``PointCloud2``. This
module turns a rendered *range image* (per-channel, per-azimuth distance +
reflectivity, with a validity mask) into that byte stream.

Supported sensors (matching :mod:`splatsim.lidar_renderer`):

* ``XT32``  — Hesai PandarXT-32. 8 blocks/packet, 32 channels/block,
  4 mm distance unit. The layout is byte-exact with the physical sensor's
  1080-byte point-cloud packet.
* ``OT128`` — Hesai Pandar OT128 (Pandar128E4X family). 2 blocks/packet,
  128 channels/block, 4 mm distance unit, with a per-block fine-azimuth byte.

Wire conventions (Hesai):

* Multi-byte integers are little-endian.
* Azimuth is encoded in units of 0.01° (``azimuth_deg * 100``), CW from the
  sensor's zero mark, wrapped to ``[0, 360)``.
* Distance is ``round(range_m / distance_unit_m)`` as ``uint16``; ``0`` means
  "no return" for that channel at that azimuth.
* Reflectivity is a ``uint8`` in ``[0, 255]``.
* The tail carries motor speed (RPM), a microsecond timestamp, the return
  mode, a factory byte (``0x42``) and a 6-byte UTC date-time.

.. note::

   The channel order written into each block follows the sensor's beam
   *table order* used by :mod:`splatsim.lidar_renderer` (row ``i`` of the
   range image -> block channel ``i``). A receiving driver reconstructs the
   point cloud with its own per-channel elevation correction, so its
   correction table must be indexed in the same order. For XT32 this is the
   natural descending-elevation order; for OT128 supply a matching
   correction file if the target firmware uses a different physical firing
   order.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ── Shared wire constants ────────────────────────────────────────────

SOP: bytes = b"\xee\xff"  # Start-of-packet marker (Hesai).
FACTORY_INFO: int = 0x42

# Return-mode byte values (Hesai). We emit single strongest return.
RETURN_MODE_STRONGEST: int = 0x37
RETURN_MODE_LAST: int = 0x38
RETURN_MODE_DUAL: int = 0x39

_UINT16_MAX = 0xFFFF


@dataclass(frozen=True)
class HesaiModel:
    """Static wire parameters for a supported Hesai LiDAR."""

    name: str
    channels: int  # lasers per block
    blocks_per_packet: int
    distance_unit_m: float  # e.g. 0.004 (4 mm)
    protocol_major: int
    protocol_minor: int
    layout: str  # "xt" | "e4x" — selects block/tail encoders
    data_port: int = 2368

    @property
    def block_size(self) -> int:
        if self.layout == "e4x":
            # azimuth(2) + fine_azimuth(1) + channels * unit(4)
            return 3 + self.channels * 4
        # xt: azimuth(2) + channels * unit(4)
        return 2 + self.channels * 4


# Registry keyed by the ``sensor_type`` string used across splatsim.
MODELS: dict[str, HesaiModel] = {
    "XT32": HesaiModel(
        name="XT32",
        channels=32,
        blocks_per_packet=8,
        distance_unit_m=0.004,
        protocol_major=0x06,
        protocol_minor=0x01,
        layout="xt",
    ),
    "OT128": HesaiModel(
        name="OT128",
        channels=128,
        blocks_per_packet=2,
        distance_unit_m=0.004,
        protocol_major=0x06,
        protocol_minor=0x01,
        layout="e4x",
    ),
}


def is_supported(sensor_type: str) -> bool:
    return sensor_type in MODELS


def get_model(sensor_type: str) -> HesaiModel:
    try:
        return MODELS[sensor_type]
    except KeyError:
        raise ValueError(
            f"Unsupported HILS LiDAR sensor_type {sensor_type!r}; "
            f"supported: {sorted(MODELS)}"
        ) from None


# ── Encoding helpers ─────────────────────────────────────────────────


def _encode_distance(
    range_m: NDArray[np.floating],
    valid: NDArray[np.bool_],
    distance_unit_m: float,
) -> NDArray[np.uint16]:
    """Range (m) -> uint16 distance in sensor units; 0 where invalid."""
    raw = np.rint(np.asarray(range_m, dtype=np.float64) / distance_unit_m)
    raw = np.clip(raw, 0, _UINT16_MAX)
    out = raw.astype(np.uint16)
    out[~valid] = 0
    return out


def _encode_reflectivity(
    intensity: NDArray[np.floating],
) -> NDArray[np.uint8]:
    """Intensity in [0, 1] -> uint8 reflectivity in [0, 255]."""
    r = np.rint(np.clip(np.asarray(intensity, dtype=np.float64), 0.0, 1.0) * 255.0)
    return r.astype(np.uint8)


def _encode_azimuth(azimuth_rad: NDArray[np.floating]) -> NDArray[np.uint16]:
    """Azimuth (rad) -> uint16 in units of 0.01°, wrapped to [0, 36000)."""
    deg = np.degrees(np.asarray(azimuth_rad, dtype=np.float64)) % 360.0
    return np.rint(deg * 100.0).astype(np.int64).astype(np.uint16)


def _fine_azimuth(azimuth_rad: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Fractional azimuth byte for E4X (1/256 of the 0.01° coarse unit)."""
    deg = np.degrees(np.asarray(azimuth_rad, dtype=np.float64)) % 360.0
    hundredths = deg * 100.0
    frac = hundredths - np.floor(hundredths)
    return np.rint(frac * 256.0).astype(np.int64).clip(0, 255).astype(np.uint8)


def _header(model: HesaiModel) -> bytes:
    """12-byte Hesai point-cloud header."""
    return struct.pack(
        "<2sBBBBBBBBBB",
        SOP,
        model.protocol_major,
        model.protocol_minor,
        0,  # reserved
        0,  # reserved
        model.channels & 0xFF,  # laser number (128 -> 0x80)
        model.blocks_per_packet,
        0,  # first-block return / echo count
        int(round(model.distance_unit_m * 1000)),  # distance unit in mm
        0,  # reserved
        0,  # reserved
    )


def _tail(
    model: HesaiModel,
    *,
    motor_speed_rpm: int,
    timestamp_us: int,
    return_mode: int,
    date_time: tuple[int, int, int, int, int, int],
    udp_sequence: int,
) -> bytes:
    """Packet tail. Layout differs slightly between the XT and E4X families.

    Both carry the fields a driver needs to timestamp and interpret the
    frame: motor speed (RPM), a microsecond timestamp, the return mode, the
    factory byte and a 6-byte UTC date-time, followed by a UDP sequence
    counter.
    """
    yy, mo, dd, hh, mi, ss = date_time
    dt = struct.pack("<6B", yy & 0xFF, mo, dd, hh, mi, ss)
    if model.layout == "e4x":
        reserved = b"\x00" * 18
    else:  # xt
        reserved = b"\x00" * 10
    return (
        reserved
        + struct.pack("<H", motor_speed_rpm & _UINT16_MAX)
        + struct.pack("<I", timestamp_us & 0xFFFFFFFF)
        + struct.pack("<B", return_mode & 0xFF)
        + struct.pack("<B", FACTORY_INFO)
        + dt
        + struct.pack("<I", udp_sequence & 0xFFFFFFFF)
    )


def _block(
    model: HesaiModel,
    azimuth_u16: int,
    fine_az: int,
    distances: NDArray[np.uint16],
    reflectivities: NDArray[np.uint8],
) -> bytes:
    """One data block: azimuth (+ fine byte for E4X) followed by units."""
    if model.layout == "e4x":
        head = struct.pack("<HB", azimuth_u16, fine_az)
        # Unit: distance(2) + reflectivity(1) + confidence(1).
        units = np.zeros((model.channels, 4), dtype=np.uint8)
        units[:, 0] = distances & 0xFF
        units[:, 1] = (distances >> 8) & 0xFF
        units[:, 2] = reflectivities
        units[:, 3] = 0  # confidence
    else:  # xt
        head = struct.pack("<H", azimuth_u16)
        # Unit: distance(2) + reflectivity(1) + reserved(1).
        units = np.zeros((model.channels, 4), dtype=np.uint8)
        units[:, 0] = distances & 0xFF
        units[:, 1] = (distances >> 8) & 0xFF
        units[:, 2] = reflectivities
        units[:, 3] = 0  # reserved
    return head + units.tobytes()


def packet_size(model: HesaiModel) -> int:
    """Total UDP payload size for one packet of ``model``."""
    header = 12
    body = model.block_size * model.blocks_per_packet
    tail = len(
        _tail(
            model,
            motor_speed_rpm=0,
            timestamp_us=0,
            return_mode=RETURN_MODE_STRONGEST,
            date_time=(0, 0, 0, 0, 0, 0),
            udp_sequence=0,
        )
    )
    return header + body + tail


def build_packets(
    model: HesaiModel,
    *,
    distance_m: NDArray[np.floating],  # (channels, n_azimuth)
    intensity: NDArray[np.floating],  # (channels, n_azimuth)
    valid: NDArray[np.bool_],  # (channels, n_azimuth)
    azimuth_rad: NDArray[np.floating],  # (n_azimuth,)
    motor_speed_rpm: int,
    timestamp_us: int,
    date_time: tuple[int, int, int, int, int, int],
    return_mode: int = RETURN_MODE_STRONGEST,
    seq_start: int = 0,
) -> list[bytes]:
    """Encode a full range image into a list of UDP data packets.

    The range image is a ``(channels, n_azimuth)`` grid where column ``a`` is
    one firing at ``azimuth_rad[a]`` and row ``c`` is beam channel ``c``.
    Columns are grouped into packets of ``model.blocks_per_packet`` blocks.

    Args:
        distance_m: Per-cell range in metres.
        intensity: Per-cell intensity in ``[0, 1]``.
        valid: Per-cell return mask; ``False`` -> encoded as no-return (0).
        azimuth_rad: Azimuth of each column (radians).
        motor_speed_rpm: Spin rate written into the tail.
        timestamp_us: Base microsecond timestamp for the frame (tail).
        date_time: UTC ``(year-1900, month, day, hour, minute, second)``.
        return_mode: Return-mode byte (default single strongest).
        seq_start: First UDP sequence number (incremented per packet).

    Returns:
        A list of packet payloads (``bytes``), in azimuth order.
    """
    if distance_m.shape != intensity.shape or distance_m.shape != valid.shape:
        raise ValueError(
            "distance_m, intensity and valid must share shape; got "
            f"{distance_m.shape}, {intensity.shape}, {valid.shape}"
        )
    channels, n_az = distance_m.shape
    if channels != model.channels:
        raise ValueError(
            f"{model.name} expects {model.channels} channels; got {channels}"
        )
    if azimuth_rad.shape[0] != n_az:
        raise ValueError(f"azimuth_rad length {azimuth_rad.shape[0]} != columns {n_az}")

    dist_u16 = _encode_distance(distance_m, valid, model.distance_unit_m)
    refl_u8 = _encode_reflectivity(intensity)
    az_u16 = _encode_azimuth(azimuth_rad)
    fine_u8 = (
        _fine_azimuth(azimuth_rad)
        if model.layout == "e4x"
        else np.zeros(n_az, dtype=np.uint8)
    )

    header = _header(model)
    bpp = model.blocks_per_packet
    packets: list[bytes] = []
    seq = seq_start
    for start in range(0, n_az, bpp):
        cols = range(start, min(start + bpp, n_az))
        blocks = bytearray()
        n_blocks = 0
        for a in cols:
            blocks += _block(
                model,
                int(az_u16[a]),
                int(fine_u8[a]),
                dist_u16[:, a],
                refl_u8[:, a],
            )
            n_blocks += 1
        # Pad a short final group up to a full packet with empty blocks so
        # every packet has the sensor's fixed size.
        if n_blocks < bpp:
            empty = _block(
                model,
                0,
                0,
                np.zeros(model.channels, dtype=np.uint16),
                np.zeros(model.channels, dtype=np.uint8),
            )
            blocks += empty * (bpp - n_blocks)
        tail = _tail(
            model,
            motor_speed_rpm=motor_speed_rpm,
            timestamp_us=timestamp_us,
            return_mode=return_mode,
            date_time=date_time,
            udp_sequence=seq,
        )
        packets.append(bytes(header + blocks + tail))
        seq += 1
    return packets


__all__ = [
    "FACTORY_INFO",
    "MODELS",
    "RETURN_MODE_DUAL",
    "RETURN_MODE_LAST",
    "RETURN_MODE_STRONGEST",
    "SOP",
    "HesaiModel",
    "build_packets",
    "get_model",
    "is_supported",
    "packet_size",
]
