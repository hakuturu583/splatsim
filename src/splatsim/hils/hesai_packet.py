"""Hesai LiDAR UDP point-cloud packet encoders for HILS.

Hardware-in-the-loop simulation feeds a real LiDAR driver (e.g. Autoware
``nebula``, ``HesaiLidar_ROS``) the exact UDP data packets that the physical
sensor would emit on the wire, instead of a decoded ``PointCloud2``. This
module turns a rendered *range image* (per-channel, per-azimuth distance +
reflectivity, with a validity mask) into that byte stream.

Encoding is done with ``torch`` tensor ops so it runs on whatever device the
range image lives on. Because the renderer produces the panorama on the GPU,
passing CUDA tensors keeps the entire encode on-device — no per-frame host
round-trip of the distance/intensity grids; only the final packed byte buffer
crosses to the host (once, for the UDP send).

.. note::

   Encoding uses ``torch.round`` where the physical wire format is defined by
   the sensor's fixed-point rounding. This can differ from a CPU reference by
   at most 1 LSB on the azimuth field (±0.01°) at floating-point tie points —
   negligible for a simulated sensor. Distance and reflectivity are exact.

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

import torch

# ── Shared wire constants ────────────────────────────────────────────

SOP: bytes = b"\xee\xff"  # Start-of-packet marker (Hesai).
FACTORY_INFO: int = 0x42

# Return-mode byte values (Hesai). We emit single strongest return.
RETURN_MODE_STRONGEST: int = 0x37
RETURN_MODE_LAST: int = 0x38
RETURN_MODE_DUAL: int = 0x39

_UINT16_MAX = 0xFFFF


# Fixed tail bytes besides the reserved prefix:
#   motor_speed(2) + timestamp(4) + return_mode(1) + factory(1)
#   + date_time(6) + udp_sequence(4) = 18.
_TAIL_FIXED = 18
_HEADER_SIZE = 12


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
    tail_reserved: int  # reserved bytes at the start of the tail
    has_fine_azimuth: bool  # per-block fractional-azimuth byte (E4X)

    @property
    def block_size(self) -> int:
        # azimuth(2) [+ fine_azimuth(1)] + channels * unit(4)
        return 2 + (1 if self.has_fine_azimuth else 0) + self.channels * 4

    @property
    def tail_size(self) -> int:
        return self.tail_reserved + _TAIL_FIXED

    @property
    def packet_size(self) -> int:
        return _HEADER_SIZE + self.block_size * self.blocks_per_packet + self.tail_size


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
        tail_reserved=10,
        has_fine_azimuth=False,
    ),
    "OT128": HesaiModel(
        name="OT128",
        channels=128,
        blocks_per_packet=2,
        distance_unit_m=0.004,
        protocol_major=0x06,
        protocol_minor=0x01,
        layout="e4x",
        tail_reserved=18,
        has_fine_azimuth=True,
    ),
}


def get_model(sensor_type: str) -> HesaiModel:
    try:
        return MODELS[sensor_type]
    except KeyError:
        raise ValueError(
            f"Unsupported HILS LiDAR sensor_type {sensor_type!r}; "
            f"supported: {sorted(MODELS)}"
        ) from None


# ── Encoding helpers ─────────────────────────────────────────────────


def _const_bytes_to_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    """A small constant byte string (header/tail) as a uint8 tensor on device."""
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


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
    return (
        b"\x00" * model.tail_reserved
        + struct.pack("<H", motor_speed_rpm & _UINT16_MAX)
        + struct.pack("<I", timestamp_us & 0xFFFFFFFF)
        + struct.pack("<B", return_mode & 0xFF)
        + struct.pack("<B", FACTORY_INFO)
        + dt
        + struct.pack("<I", udp_sequence & 0xFFFFFFFF)
    )


def build_frame_tensor(
    model: HesaiModel,
    *,
    distance_m: torch.Tensor,  # (channels, n_azimuth) float, metres
    intensity: torch.Tensor,  # (channels, n_azimuth) float, [0, 1]
    valid: torch.Tensor,  # (channels, n_azimuth) bool
    azimuth_rad: torch.Tensor,  # (n_azimuth,) float, radians
    motor_speed_rpm: int,
    timestamp_us: int,
    date_time: tuple[int, int, int, int, int, int],
    return_mode: int = RETURN_MODE_STRONGEST,
    seq_start: int = 0,
) -> torch.Tensor:
    """Encode a range image into a ``(n_packets, packet_size)`` uint8 tensor.

    The range image is a ``(channels, n_azimuth)`` grid where column ``a`` is
    one firing at ``azimuth_rad[a]`` and row ``c`` is beam channel ``c``.
    Columns are grouped into packets of ``model.blocks_per_packet`` blocks.

    All arithmetic runs with ``torch`` on the input tensors' device, so CUDA
    inputs are encoded on the GPU and the result stays on the GPU until the
    caller moves it to the host. The buffer is row-contiguous, so each row is a
    ready-to-send packet once on the host (``row.tobytes()`` /
    ``socket.sendto(row, ...)``). The header and tail are written once and
    broadcast across every packet row; only the per-packet UDP sequence varies.

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
    """
    if distance_m.shape != intensity.shape or distance_m.shape != valid.shape:
        raise ValueError(
            "distance_m, intensity and valid must share shape; got "
            f"{tuple(distance_m.shape)}, {tuple(intensity.shape)}, "
            f"{tuple(valid.shape)}"
        )
    channels, n_az = distance_m.shape
    if channels != model.channels:
        raise ValueError(
            f"{model.name} expects {model.channels} channels; got {channels}"
        )
    if azimuth_rad.shape[0] != n_az:
        raise ValueError(f"azimuth_rad length {azimuth_rad.shape[0]} != columns {n_az}")

    device = distance_m.device
    bpp = model.blocks_per_packet
    n_pad = ((n_az + bpp - 1) // bpp) * bpp  # round up to a whole packet count
    n_packets = n_pad // bpp
    block_size = model.block_size
    psize = model.packet_size
    tail_off = _HEADER_SIZE + bpp * block_size

    # ── per-cell fixed-point encode (int32 intermediates: torch's uint16
    # arithmetic is immature, so compute in int32 and narrow the byte lanes) ──
    raw = torch.clamp(
        torch.round(distance_m / model.distance_unit_m), 0, _UINT16_MAX
    ).to(torch.int32)
    raw = raw * valid.to(torch.int32)  # zero the no-return cells
    dist_lo = (raw & 0xFF).to(torch.uint8)  # (channels, n_az)
    dist_hi = ((raw >> 8) & 0xFF).to(torch.uint8)
    refl = torch.clamp(
        torch.round(torch.clamp(intensity, 0.0, 1.0) * 255.0), 0, 255
    ).to(torch.uint8)

    deg = torch.remainder(torch.rad2deg(azimuth_rad), 360.0)  # (n_az,)
    az = torch.clamp(torch.round(deg * 100.0), 0, _UINT16_MAX).to(torch.int32)

    # ── body grid (n_pad, block_size); pad rows stay zero for fixed size ──
    body = torch.zeros((n_pad, block_size), dtype=torch.uint8, device=device)
    body[:n_az, 0] = (az & 0xFF).to(torch.uint8)
    body[:n_az, 1] = ((az >> 8) & 0xFF).to(torch.uint8)
    unit_off = 2
    if model.has_fine_azimuth:
        frac = (deg * 100.0) - torch.floor(deg * 100.0)
        body[:n_az, 2] = torch.clamp(torch.round(frac * 256.0), 0, 255).to(torch.uint8)
        unit_off = 3

    # Units: distance_lo(1) + distance_hi(1) + reflectivity(1) + reserved(1).
    units = torch.zeros((n_az, channels, 4), dtype=torch.uint8, device=device)
    units[..., 0] = dist_lo.T
    units[..., 1] = dist_hi.T
    units[..., 2] = refl.T
    body[:n_az, unit_off : unit_off + channels * 4] = units.reshape(n_az, channels * 4)

    # ── assemble packets ──
    buf = torch.zeros((n_packets, psize), dtype=torch.uint8, device=device)
    buf[:, :_HEADER_SIZE] = _const_bytes_to_tensor(_header(model), device)
    buf[:, _HEADER_SIZE:tail_off] = body.reshape(n_packets, bpp * block_size)
    tail = _tail(
        model,
        motor_speed_rpm=motor_speed_rpm,
        timestamp_us=timestamp_us,
        return_mode=return_mode,
        date_time=date_time,
        udp_sequence=0,
    )
    buf[:, tail_off:] = _const_bytes_to_tensor(tail, device)
    # Per-packet UDP sequence: last 4 bytes, little-endian u32.
    seqs = (seq_start + torch.arange(n_packets, device=device, dtype=torch.int64)) & (
        0xFFFFFFFF
    )
    for b in range(4):
        buf[:, psize - 4 + b] = ((seqs >> (8 * b)) & 0xFF).to(torch.uint8)
    return buf


def build_packets(
    model: HesaiModel,
    *,
    distance_m: torch.Tensor,
    intensity: torch.Tensor,
    valid: torch.Tensor,
    azimuth_rad: torch.Tensor,
    motor_speed_rpm: int,
    timestamp_us: int,
    date_time: tuple[int, int, int, int, int, int],
    return_mode: int = RETURN_MODE_STRONGEST,
    seq_start: int = 0,
) -> list[bytes]:
    """Encode a range image into a list of UDP packet payloads (``bytes``).

    Thin ``bytes`` wrapper over :func:`build_frame_tensor` for callers that
    want individual payloads (tests, non-socket use). The hot UDP path should
    prefer :func:`build_frame_tensor` and send its rows directly. Args mirror
    :func:`build_frame_tensor`.
    """
    buf = build_frame_tensor(
        model,
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=azimuth_rad,
        motor_speed_rpm=motor_speed_rpm,
        timestamp_us=timestamp_us,
        date_time=date_time,
        return_mode=return_mode,
        seq_start=seq_start,
    )
    return [row.tobytes() for row in buf.cpu().numpy()]


__all__ = [
    "FACTORY_INFO",
    "MODELS",
    "RETURN_MODE_DUAL",
    "RETURN_MODE_LAST",
    "RETURN_MODE_STRONGEST",
    "SOP",
    "HesaiModel",
    "build_frame_tensor",
    "build_packets",
    "get_model",
]
