"""Round-trip and wire-format tests for the Hesai HILS packet encoders.

Encoding runs on ``torch`` tensors (CPU here; CUDA in production). Distance and
reflectivity are exact; azimuth may differ from an ideal fixed-point reference
by at most 1 LSB (±0.01°) at floating-point tie points, so azimuth is checked
with a ±1 tolerance.
"""

from __future__ import annotations

import math
import socket
import struct

import numpy as np
import pytest
import torch

from splatsim.hils import HesaiHilsPublisher, build_packets, get_model
from splatsim.hils.hesai_packet import (
    FACTORY_INFO,
    RETURN_MODE_STRONGEST,
    SOP,
    build_frame_tensor,
)


def _decode_packet(model, payload: bytes) -> dict:
    """Minimal decoder mirroring the encoder, for round-trip verification."""
    assert len(payload) == model.packet_size
    assert payload[:2] == SOP
    laser_num = payload[6]
    block_num = payload[7]
    dis_unit_mm = payload[9]
    assert laser_num == (model.channels & 0xFF)
    assert block_num == model.blocks_per_packet

    off = 12
    blocks = []
    for _ in range(model.blocks_per_packet):
        if model.layout == "e4x":
            (az,) = struct.unpack_from("<H", payload, off)
            fine = payload[off + 2]
            off += 3
        else:
            (az,) = struct.unpack_from("<H", payload, off)
            fine = 0
            off += 2
        units = np.frombuffer(
            payload, dtype=np.uint8, count=model.channels * 4, offset=off
        ).reshape(model.channels, 4)
        off += model.channels * 4
        dist = units[:, 0].astype(np.uint16) | (units[:, 1].astype(np.uint16) << 8)
        refl = units[:, 2].copy()
        blocks.append((az, fine, dist, refl))

    tail = payload[off:]
    # Tail layout: reserved | motor_speed(2) | timestamp(4) | return_mode(1)
    #            | factory(1) | date_time(6) | seq(4)
    reserved_len = 18 if model.layout == "e4x" else 10
    t = reserved_len
    (motor_speed,) = struct.unpack_from("<H", tail, t)
    t += 2
    (timestamp_us,) = struct.unpack_from("<I", tail, t)
    t += 4
    return_mode = tail[t]
    factory = tail[t + 1]
    date_time = tuple(tail[t + 2 : t + 8])
    (seq,) = struct.unpack_from("<I", tail, t + 8)

    return {
        "dis_unit_mm": dis_unit_mm,
        "blocks": blocks,
        "motor_speed": motor_speed,
        "timestamp_us": timestamp_us,
        "return_mode": return_mode,
        "factory": factory,
        "date_time": date_time,
        "seq": seq,
    }


def _azimuth_close(actual: int, expected: int, tol: int = 1) -> bool:
    """True if two 0.01°-unit azimuths are within ``tol`` LSB (circular)."""
    diff = abs(int(actual) - int(expected)) % 36000
    return min(diff, 36000 - diff) <= tol


def _make_frame(model, n_az: int):
    """Build a deterministic range image as CPU torch tensors."""
    channels = model.channels
    rng = np.arange(channels * n_az, dtype=np.float64).reshape(channels, n_az)
    # Distances as exact multiples of the distance unit so quantization is lossless.
    distance_m = (rng % 100 + 1) * model.distance_unit_m * 250.0  # up to ~99 m
    intensity = ((rng % 256) / 255.0).astype(np.float32)
    valid = np.ones((channels, n_az), dtype=bool)
    valid[0, 0] = False  # one no-return cell
    azimuth_rad = np.linspace(math.pi, -math.pi, n_az, endpoint=False)
    return (
        torch.from_numpy(distance_m.astype(np.float32)),
        torch.from_numpy(intensity),
        torch.from_numpy(valid),
        torch.from_numpy(azimuth_rad.astype(np.float32)),
    )


@pytest.mark.parametrize("sensor_type", ["XT32", "OT128"])
def test_packet_size_and_count(sensor_type: str) -> None:
    model = get_model(sensor_type)
    n_az = model.blocks_per_packet * 3 + 1  # not a whole multiple
    distance_m, intensity, valid, az = _make_frame(model, n_az)
    packets = build_packets(
        model,
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        motor_speed_rpm=600,
        timestamp_us=123456,
        date_time=(125, 7, 10, 1, 2, 3),
    )
    expected_count = math.ceil(n_az / model.blocks_per_packet)
    assert len(packets) == expected_count
    for pkt in packets:
        assert len(pkt) == model.packet_size


def test_xt32_packet_size_is_1080() -> None:
    assert get_model("XT32").packet_size == 1080


@pytest.mark.parametrize("sensor_type", ["XT32", "OT128"])
def test_round_trip_values(sensor_type: str) -> None:
    model = get_model(sensor_type)
    n_az = model.blocks_per_packet  # exactly one packet
    distance_m, intensity, valid, az = _make_frame(model, n_az)
    packets = build_packets(
        model,
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        motor_speed_rpm=600,
        timestamp_us=777,
        date_time=(125, 7, 10, 1, 2, 3),
        return_mode=RETURN_MODE_STRONGEST,
        seq_start=5,
    )
    assert len(packets) == 1
    dec = _decode_packet(model, packets[0])

    assert dec["dis_unit_mm"] == round(model.distance_unit_m * 1000)
    assert dec["return_mode"] == RETURN_MODE_STRONGEST
    assert dec["factory"] == FACTORY_INFO
    assert dec["motor_speed"] == 600
    assert dec["timestamp_us"] == 777
    assert dec["date_time"] == (125, 7, 10, 1, 2, 3)
    assert dec["seq"] == 5

    distance_np = distance_m.numpy()
    intensity_np = intensity.numpy()
    valid_np = valid.numpy()
    az_np = az.numpy()
    for a, (az_u16, _fine, dist, refl) in enumerate(dec["blocks"]):
        # Azimuth round-trips within one 0.01deg unit (torch.round tie-breaking).
        expected_az = round((math.degrees(az_np[a]) % 360.0) * 100.0) % 36000
        assert _azimuth_close(az_u16, expected_az)
        # Distances round-trip to metres (chosen as exact unit multiples).
        decoded_m = dist.astype(np.float64) * model.distance_unit_m
        for c in range(model.channels):
            if valid_np[c, a]:
                assert decoded_m[c] == pytest.approx(distance_np[c, a], abs=1e-4)
            else:
                assert dist[c] == 0  # no-return
        # Reflectivity round-trips to the nearest 1/255.
        expected_refl = np.rint(np.clip(intensity_np[:, a], 0, 1) * 255).astype(
            np.uint8
        )
        assert np.array_equal(refl, expected_refl)


def test_invalid_cells_are_no_return() -> None:
    model = get_model("XT32")
    n_az = model.blocks_per_packet
    distance_m, intensity, valid, az = _make_frame(model, n_az)
    valid = torch.zeros_like(valid)  # everything drops
    packets = build_packets(
        model,
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        motor_speed_rpm=600,
        timestamp_us=0,
        date_time=(0, 1, 1, 0, 0, 0),
    )
    dec = _decode_packet(model, packets[0])
    for _az, _fine, dist, _refl in dec["blocks"]:
        assert np.all(dist == 0)


def test_publisher_sends_over_udp() -> None:
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    recv.settimeout(2.0)
    host, port = recv.getsockname()

    # Pin the sim start epoch so the packet date-time is deterministic.
    pub = HesaiHilsPublisher(
        "XT32", host=host, port=port, start_epoch_s=1_700_000_000.0
    )
    model = pub.model
    n_az = model.blocks_per_packet
    distance_m, intensity, valid, az = _make_frame(model, n_az)

    n_sent = pub.publish(
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        spin_hz=10.0,
        sim_time_s=0.5,  # start_epoch + 0.5 s -> 500_000 us sub-second
    )
    assert n_sent == 1

    payload, _addr = recv.recvfrom(4096)
    dec = _decode_packet(model, payload)
    assert dec["motor_speed"] == 600  # 10 Hz * 60
    assert dec["timestamp_us"] == 500_000  # 0.5 s sub-second part
    pub.close()
    recv.close()


@pytest.mark.parametrize("sensor_type", ["XT32", "OT128"])
def test_frame_tensor_rows_match_packets(sensor_type: str) -> None:
    """The encoded byte grid's rows equal the list-of-bytes form."""
    model = get_model(sensor_type)
    n_az = model.blocks_per_packet * 4 + 1  # exercises the pad row
    distance_m, intensity, valid, az = _make_frame(model, n_az)
    kwargs = dict(
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        motor_speed_rpm=600,
        timestamp_us=99,
        date_time=(125, 7, 10, 1, 2, 3),
        seq_start=3,
    )
    buf = build_frame_tensor(model, **kwargs)
    packets = build_packets(model, **kwargs)
    assert tuple(buf.shape) == (len(packets), model.packet_size)
    assert buf.dtype == torch.uint8
    rows = buf.cpu().numpy()
    for k, pkt in enumerate(packets):
        assert rows[k].tobytes() == pkt


def test_unsupported_sensor_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported HILS LiDAR"):
        get_model("VLP16")
