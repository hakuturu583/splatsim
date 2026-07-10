"""Round-trip and wire-format tests for the Hesai HILS packet encoders."""

from __future__ import annotations

import math
import socket
import struct

import numpy as np
import pytest

from splatsim.hils import HesaiHilsPublisher, build_packets, get_model, packet_size
from splatsim.hils.hesai_packet import (
    FACTORY_INFO,
    RETURN_MODE_STRONGEST,
    SOP,
)


def _decode_packet(model, payload: bytes) -> dict:
    """Minimal decoder mirroring the encoder, for round-trip verification."""
    assert len(payload) == packet_size(model)
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


def _make_frame(model, n_az: int):
    channels = model.channels
    rng = np.arange(channels * n_az, dtype=np.float64).reshape(channels, n_az)
    # Distances as exact multiples of the distance unit so quantization is lossless.
    distance_m = (rng % 100 + 1) * model.distance_unit_m * 250.0  # up to ~99 m
    intensity = ((rng % 256) / 255.0).astype(np.float32)
    valid = np.ones((channels, n_az), dtype=bool)
    valid[0, 0] = False  # one no-return cell
    azimuth_rad = np.linspace(math.pi, -math.pi, n_az, endpoint=False)
    return distance_m, intensity, valid, azimuth_rad


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
        assert len(pkt) == packet_size(model)


def test_xt32_packet_size_is_1080() -> None:
    assert packet_size(get_model("XT32")) == 1080


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

    for a, (az_u16, _fine, dist, refl) in enumerate(dec["blocks"]):
        # Azimuth round-trips within one 0.01deg unit.
        expected_az = round((math.degrees(az[a]) % 360.0) * 100.0) % 36000
        assert az_u16 == expected_az
        # Distances round-trip to metres (chosen as exact unit multiples).
        decoded_m = dist.astype(np.float64) * model.distance_unit_m
        for c in range(model.channels):
            if valid[c, a]:
                assert decoded_m[c] == pytest.approx(distance_m[c, a], abs=1e-6)
            else:
                assert dist[c] == 0  # no-return
        # Reflectivity round-trips to the nearest 1/255.
        expected_refl = np.rint(np.clip(intensity[:, a], 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(refl, expected_refl)


def test_invalid_cells_are_no_return() -> None:
    model = get_model("XT32")
    n_az = model.blocks_per_packet
    distance_m, intensity, valid, az = _make_frame(model, n_az)
    valid[:] = False  # everything drops
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

    pub = HesaiHilsPublisher("XT32", host=host, port=port)
    model = pub.model
    n_az = model.blocks_per_packet
    distance_m, intensity, valid, az = _make_frame(model, n_az)

    n_sent = pub.publish(
        distance_m=distance_m,
        intensity=intensity,
        valid=valid,
        azimuth_rad=az,
        spin_hz=10.0,
        epoch_s=1_700_000_000.5,
    )
    assert n_sent == 1

    payload, _addr = recv.recvfrom(4096)
    dec = _decode_packet(model, payload)
    assert dec["motor_speed"] == 600  # 10 Hz * 60
    assert dec["timestamp_us"] == 500_000  # 0.5 s sub-second part
    pub.close()
    recv.close()


def test_unsupported_sensor_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported HILS LiDAR"):
        get_model("VLP16")
