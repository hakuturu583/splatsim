"""Tests for reading rigid dynamic-object ("actor") assets out of a scene USDZ.

The bank (3dgs_io ``splatsim.actor_assets/v1``) carries object-local Gaussian
clouds; splatsim turns them into posable :class:`RigidBody` instances that a
scenario drives from outside. What is worth defending here is what a consumer
cannot recover on its own: that an asset's Gaussians and its per-Gaussian LiDAR
attributes come through the same readers a background chunk uses, that a
world-frame pose lands in the renderer's tile-local frame, and that posing an
actor re-expresses its view-dependent colour instead of spinning the highlights
with the car.
"""

from __future__ import annotations

import importlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from splatsim._conversions import (
    MAX_NON_YAW_RAD,
    GaussianTensors,
    apply_rigid_transform,
    rotate_sh_about_z,
    yaw_from_quat,
)
from splatsim.actor_assets import (
    ActorAssetLibrary,
    has_actor_assets,
    pose_from_track_frame,
    world_to_tile_local,
)

_3dgs_io = importlib.import_module("3dgs_io")
_spz = importlib.import_module("spz")
_spz_io = importlib.import_module("3dgs_io.spz_io")

FRAME_CONVENTION = _3dgs_io.FRAME_CONVENTION
ActorAssetSource = _3dgs_io.ActorAssetSource
ActorInstance = _3dgs_io.ActorInstance
build_actor_asset_bank = _3dgs_io.build_actor_asset_bank
serialize_actor_assets = _3dgs_io.serialize_actor_assets

CPU = torch.device("cpu")
CAR_SIZE = (4.5, 1.9, 1.5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _object_local_cloud(n: int = 32, *, seed: int = 0, sh_degree: int = 0):
    """A car-shaped cloud in the object frame (centred on the origin)."""
    rng = np.random.default_rng(seed)
    gc = _spz.GaussianCloud()  # ty: ignore[unresolved-attribute]
    gc.antialiased = False
    box = (rng.random((n, 3)) - 0.5) * np.array(CAR_SIZE)
    gc.positions = box.astype(np.float32).reshape(-1)
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.astype(np.float32).reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.0, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    per_ch = (sh_degree + 1) ** 2 - 1
    if per_ch:
        gc.sh_degree = sh_degree
        gc.sh = rng.standard_normal(n * per_ch * 3).astype(np.float32)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _source(
    asset_id: str = "sedan_0007",
    *,
    n: int = 32,
    seed: int = 0,
    with_lidar: bool = False,
    sh_degree: int = 0,
) -> Any:
    rng = np.random.default_rng(seed + 7)
    ext: dict[str, np.ndarray] = {}
    if with_lidar:
        ext = {
            "lidar_intensity_raw": np.linspace(-1.0, 1.0, n).astype(np.float32),
            "lidar_raydrop_logit": np.full(n, -0.5, dtype=np.float32),
            "lidar_mask": (rng.random(n) > 0.5).astype(np.float32),
        }
    return ActorAssetSource(
        asset_id=asset_id,
        cloud=_object_local_cloud(n, seed=seed, sh_degree=sh_degree),
        class_name="automobile",
        size=CAR_SIZE,
        ext_attrs=ext,
    )


def _real_spz_chunk(n: int = 4) -> bytes:
    """Real NGSP v4 SPZ bytes for a tiny world-frame background cloud."""
    rng = np.random.default_rng(0)
    gc = _spz.GaussianCloud()  # ty: ignore[unresolved-attribute]
    gc.antialiased = False
    gc.positions = rng.uniform(-5.0, 5.0, size=n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.0, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)
    return _spz_io.save_spz_world_bytes(gc)


def _write_scene_usdz(
    path: Path,
    sources: list[Any],
    instances: list[Any] | None = None,
    *,
    n_background: int = 4,
) -> Path:
    """A minimal scene/v3 bundle carrying an actor asset bank."""
    bank, payloads = build_actor_asset_bank(sources, instances or [])
    scene: dict[str, Any] = {
        "schema": "splatsim.scene/v3",
        "world": {
            "frame_convention": FRAME_CONVENTION,
            "ecef_anchor": np.eye(4).tolist(),
        },
        "gaussians": {
            "frame": "world",
            "chunks": [{"uri": "chunks/chunk_000000.spz", "n_points": n_background}],
        },
        "extras": {"actor_assets": "actor_assets.json"},
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))
        zf.writestr("chunks/chunk_000000.spz", _real_spz_chunk(n_background))
        zf.writestr("actor_assets.json", json.dumps(serialize_actor_assets(bank)))
        for asset in bank.assets:
            zf.writestr(str(asset.uri), payloads[asset.asset_id])
    return path


# ---------------------------------------------------------------------------
# Loading the bank
# ---------------------------------------------------------------------------


def test_library_reads_every_asset_and_its_index_metadata(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz",
        [_source("sedan_0007", n=32), _source("truck_0001", n=16, seed=2)],
    )
    library = ActorAssetLibrary(usdz, device=CPU)

    assert len(library) == 2
    assert sorted(library.asset_ids) == ["sedan_0007", "truck_0001"]
    assert "sedan_0007" in library

    info = library.info("sedan_0007")
    assert info.class_name == "automobile"
    assert info.n_points == 32
    assert info.size == pytest.approx(CAR_SIZE)
    assert info.has_lidar_attributes is False


def test_library_reports_an_unknown_asset(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    with pytest.raises(KeyError, match="no actor asset 'missing'"):
        library.spawn("missing")


def test_library_rejects_a_bundle_without_a_bank(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    # Strip the extras pointer the way an older bundle would have it.
    stripped = tmp_path / "no_bank.usdz"
    with zipfile.ZipFile(usdz) as src, zipfile.ZipFile(stripped, "w") as dst:
        for name in src.namelist():
            if name == "scene.json":
                scene = json.loads(src.read(name))
                scene["extras"] = {}
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))

    assert has_actor_assets(usdz) is True
    assert has_actor_assets(stripped) is False
    with pytest.raises(ValueError, match="no actor asset bank"):
        ActorAssetLibrary(stripped, device=CPU)


def test_lidar_attributes_come_through_the_background_reader(tmp_path: Path) -> None:
    """An actor's per-Gaussian LiDAR attrs are read exactly like a chunk's."""
    n = 32
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source(n=n, with_lidar=True)])
    library = ActorAssetLibrary(usdz, device=CPU)
    assert library.info("sedan_0007").has_lidar_attributes is True

    body = library.spawn("sedan_0007")
    base = body.base_tensors
    assert base.intensity_raw is not None
    assert base.raydrop_logit is not None
    torch.testing.assert_close(
        base.raydrop_logit, torch.full((n,), -0.5), atol=0.02, rtol=0
    )
    assert base.lidar_mask is not None
    assert base.lidar_mask.dtype == torch.bool


def test_bound_track_ids_are_exposed_but_do_not_drive_poses(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(
        tmp_path / "scene.usdz",
        [_source("sedan_0007")],
        [
            ActorInstance(track_id="100", asset_id="sedan_0007"),
            ActorInstance(track_id="101", asset_id="sedan_0007"),
        ],
    )
    library = ActorAssetLibrary(usdz, device=CPU)
    assert library.bound_track_ids("sedan_0007") == ["100", "101"]
    # Spawning is independent of the bindings: the caller decides.
    assert library.spawn("sedan_0007").position.tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Spawning and posing
# ---------------------------------------------------------------------------


def test_instances_share_their_base_tensors(tmp_path: Path) -> None:
    """Fifty of the same sedan must cost one upload, not fifty."""
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    a = library.spawn("sedan_0007")
    b = library.spawn("sedan_0007")
    assert a is not b
    assert a.base_tensors.means.data_ptr() == b.base_tensors.means.data_ptr()


def test_spawn_applies_the_initial_pose(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    yaw = math.radians(90.0)
    body = library.spawn(
        "sedan_0007",
        position=(10.0, -3.0, 0.75),
        rotation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    )
    assert body.position.tolist() == pytest.approx([10.0, -3.0, 0.75])

    # A +90 degrees yaw sends the car's own +x to world +y, so the posed means
    # are the object-local ones through R_z(90) plus the translation.
    local = body.base_tensors.means
    turned = local @ torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        body.tensors.means - body.position, turned, atol=1e-5, rtol=0
    )


class _FakeBackground:
    """A stand-in for the one attribute :class:`TileLocalFrame` asks for."""

    def __init__(self, centroid: tuple[float, float, float]) -> None:
        self.tile_local_centroid = torch.tensor(centroid, dtype=torch.float32)


def test_world_positions_are_recentred_into_the_tile_local_frame(
    tmp_path: Path,
) -> None:
    """A world-frame pose must have the background's centroid subtracted.

    Without it a car tracked at ENU (113, -58, 1.9) renders hundreds of metres
    from the scene, because Background re-centres its cloud on load.
    """
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    background = _FakeBackground((100.0, -50.0, 1.0))

    world = (113.62, -58.55, 1.92)
    torch.testing.assert_close(
        world_to_tile_local(world, background),
        torch.tensor([13.62, -8.55, 0.92]),
        atol=1e-4,
        rtol=0,
    )

    body = library.spawn("sedan_0007", position=world, background=background)
    torch.testing.assert_close(
        body.position, torch.tensor([13.62, -8.55, 0.92]), atol=1e-4, rtol=0
    )

    # Without a background the position is taken as already tile-local.
    plain = library.spawn("sedan_0007", position=world)
    torch.testing.assert_close(plain.position, torch.tensor(world), atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# View-dependent colour
# ---------------------------------------------------------------------------

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def _sh_basis(d: np.ndarray) -> np.ndarray:
    """The 3DGS real-SH basis, DC first — the layout ``colors`` uses."""
    x, y, z = d
    return np.array(
        [
            _SH_C0,
            _SH_C1 * -y,
            _SH_C1 * z,
            _SH_C1 * -x,
            _SH_C2[0] * x * y,
            _SH_C2[1] * y * z,
            _SH_C2[2] * (2 * z * z - x * x - y * y),
            _SH_C2[3] * x * z,
            _SH_C2[4] * (x * x - y * y),
            _SH_C3[0] * y * (3 * x * x - y * y),
            _SH_C3[1] * x * y * z,
            _SH_C3[2] * y * (4 * z * z - x * x - y * y),
            _SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
            _SH_C3[4] * x * (4 * z * z - x * x - y * y),
            _SH_C3[5] * z * (x * x - y * y),
            _SH_C3[6] * x * (x * x - 3 * y * y),
        ]
    )


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, -117.0, 180.0])
def test_sh_z_rotation_is_exact(yaw_deg: float) -> None:
    """``f_world(d) == f_object(R_z(yaw) @ d)`` for every direction."""
    rng = np.random.default_rng(5)
    colors = torch.tensor(rng.standard_normal((2, 16, 3)), dtype=torch.float64)
    yaw = math.radians(yaw_deg)
    rotated = rotate_sh_about_z(colors, torch.tensor(yaw, dtype=torch.float64))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for _ in range(20):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        np.testing.assert_allclose(
            _sh_basis(d) @ rotated[0].numpy(),
            _sh_basis(rot @ d) @ colors[0].numpy(),
            atol=1e-12,
        )


def test_sh_z_rotation_leaves_the_dc_band_alone() -> None:
    """Band 0 is rotation-invariant, so a view-independent asset is untouched."""
    colors = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    rotated = rotate_sh_about_z(colors, torch.tensor(1.234))
    torch.testing.assert_close(rotated[:, 0, :], colors[:, 0, :])
    assert not torch.allclose(rotated[:, 1:, :], colors[:, 1:, :])


def _tensors_with_sh(n: int = 8, k: int = 16) -> GaussianTensors:
    rng = np.random.default_rng(11)
    return GaussianTensors(
        means=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        scales=torch.ones(n, 3),
        opacities=torch.ones(n),
        colors=torch.tensor(rng.standard_normal((n, k, 3)), dtype=torch.float32),
        sh_degree=3,
    )


def test_rigid_transform_rotates_sh_only_when_asked() -> None:
    """Existing rigid bodies keep their behaviour; actors opt in."""
    base = _tensors_with_sh()
    yaw = math.radians(40.0)
    rotation = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    position = torch.zeros(3)

    plain = apply_rigid_transform(base, position, rotation)
    torch.testing.assert_close(plain.colors, base.colors)

    turned = apply_rigid_transform(base, position, rotation, rotate_sh=True)
    assert not torch.allclose(turned.colors, base.colors)
    torch.testing.assert_close(
        turned.colors, rotate_sh_about_z(base.colors, torch.tensor(yaw))
    )


def test_rgb_mode_is_unaffected_by_rotate_sh() -> None:
    """An asset loaded without SH has [N, 3] colors and nothing to rotate."""
    base = GaussianTensors(
        means=torch.zeros(4, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
        scales=torch.ones(4, 3),
        opacities=torch.ones(4),
        colors=torch.rand(4, 3),
        sh_degree=0,
    )
    out = apply_rigid_transform(
        base, torch.zeros(3), torch.tensor([0.7, 0.0, 0.0, 0.714]), rotate_sh=True
    )
    torch.testing.assert_close(out.colors, base.colors)


def test_spawned_actors_rotate_their_sh(tmp_path: Path) -> None:
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source(sh_degree=3)])
    library = ActorAssetLibrary(usdz, device=CPU, use_sh=True)
    body = library.spawn("sedan_0007")
    assert body.rotate_sh is True

    yaw = math.radians(55.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    assert not torch.allclose(body.tensors.colors, body.base_tensors.colors)


def test_a_tilted_pose_reports_that_its_sh_rotation_is_approximate() -> None:
    """Yaw-only SH rotation is exact for road vehicles and says so when it isn't."""
    from splatsim.rigid_body import RigidBody

    body = RigidBody.from_tensors(_tensors_with_sh(), device=CPU, rotate_sh=True)
    yaw = math.radians(30.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    assert body.sh_rotation_is_exact
    assert body.sh_rotation_tilt == pytest.approx(0.0, abs=1e-6)

    pitch = math.radians(20.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0))
    assert not body.sh_rotation_is_exact
    assert body.sh_rotation_tilt == pytest.approx(pitch, abs=1e-5)
    assert body.sh_rotation_tilt > MAX_NON_YAW_RAD


def test_a_body_without_sh_rotation_reports_no_approximation() -> None:
    from splatsim.rigid_body import RigidBody

    body = RigidBody.from_tensors(_tensors_with_sh(), device=CPU)
    pitch = math.radians(20.0)
    body.set_pose((0.0, 0.0, 0.0), (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0))
    assert body.rotate_sh is False
    assert body.sh_rotation_tilt == 0.0
    assert body.sh_rotation_is_exact


def test_track_poses_are_reordered_from_xyzw_to_wxyz() -> None:
    """Bundle poses are xyzw; splatsim is wxyz. Getting that wrong is silent."""
    frame = _3dgs_io.TrackFrame(
        timestamp_us=1_000_000,
        translation=(113.62, -58.55, 1.92),
        rotation=(0.1, 0.2, 0.3, 0.927),  # xyzw
    )
    position, rotation = pose_from_track_frame(frame)
    assert position == pytest.approx((113.62, -58.55, 1.92))
    assert rotation == pytest.approx((0.927, 0.1, 0.2, 0.3))


def test_spawning_from_a_track_pose_lands_on_the_tracked_box(tmp_path: Path) -> None:
    """The documented path: bundle pose -> wxyz -> tile-local -> posed body."""
    usdz = _write_scene_usdz(tmp_path / "scene.usdz", [_source()])
    library = ActorAssetLibrary(usdz, device=CPU)
    background = _FakeBackground((100.0, -50.0, 1.0))

    yaw = math.radians(33.0)
    frame = _3dgs_io.TrackFrame(
        timestamp_us=1_000_000,
        translation=(113.62, -58.55, 1.92),
        rotation=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),  # xyzw
    )
    position, rotation = pose_from_track_frame(frame)
    body = library.spawn(
        "sedan_0007", position=position, rotation=rotation, background=background
    )

    torch.testing.assert_close(
        body.position, torch.tensor([13.62, -8.55, 0.92]), atol=1e-4, rtol=0
    )
    # Back out of the pose: every gaussian returns to the object-local box.
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    back = (body.tensors.means - body.position) @ rot
    half = torch.tensor(CAR_SIZE) / 2
    assert bool((back.abs() <= half + 1e-4).all())


def test_yaw_from_quat_splits_heading_from_tilt() -> None:
    yaw = math.radians(33.0)
    got_yaw, tilt = yaw_from_quat(
        torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    )
    assert float(got_yaw) == pytest.approx(yaw, abs=1e-6)
    assert float(tilt) == pytest.approx(0.0, abs=1e-6)
