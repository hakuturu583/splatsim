"""Tests for gRPC LoD control (Initialize default + SetLod runtime toggle).

These exercise the servicer's LoD state handling directly, without loading a
scene onto a GPU: a :class:`Scene` built with a ``LodManager`` but no
background is enough to drive ``lod_enabled``.
"""

from __future__ import annotations

from typing import cast

import grpc

from splatsim.dataclass.lod_config import LodConfig
from splatsim.grpc_service._generated import rendering_service_pb2 as pb2
from splatsim.grpc_service.server import RenderingServiceServicer
from splatsim.lod import LodManager
from splatsim.scene import Scene

# SetLod never touches the ServicerContext, so a typed None stand-in is enough
# to drive the handler in these unit tests.
_CTX = cast(grpc.ServicerContext, None)


def test_enable_lod_is_optional_with_presence() -> None:
    """``enable_lod`` must keep proto3 presence.

    Initialize distinguishes an unset field (LoD stays on by default) from an
    explicit ``False`` (start with LoD off) via ``HasField``. Dropping
    ``optional`` in the proto would make that check meaningless and silently
    break the default-on behavior in the Docker/gRPC path.
    """
    req = pb2.InitializeRequest()
    assert req.HasField("enable_lod") is False

    req.enable_lod = False
    assert req.HasField("enable_lod") is True
    assert req.enable_lod is False


def test_setlod_not_initialized() -> None:
    """SetLod before Initialize fails cleanly rather than crashing."""
    servicer = RenderingServiceServicer()
    resp = servicer.SetLod(pb2.SetLodRequest(enabled=True), _CTX)
    assert resp.success is False
    assert resp.enabled is False
    assert "Initialize" in resp.message


def test_setlod_toggles_when_manager_present() -> None:
    """With a LoD-capable scene, SetLod flips ``scene.lod_enabled`` both ways."""
    servicer = RenderingServiceServicer()
    scene = Scene(background=None, lod_manager=LodManager(LodConfig()))
    servicer._scene = scene
    servicer._initialized = True

    off = servicer.SetLod(pb2.SetLodRequest(enabled=False), _CTX)
    assert off.success is True
    assert off.enabled is False
    assert scene.lod_enabled is False

    on = servicer.SetLod(pb2.SetLodRequest(enabled=True), _CTX)
    assert on.success is True
    assert on.enabled is True
    assert scene.lod_enabled is True
    assert on.message == ""


def test_setlod_enable_without_manager_reports_unavailable() -> None:
    """Enabling LoD on a scene with no manager stays off and reports why."""
    servicer = RenderingServiceServicer()
    scene = Scene(background=None, lod_manager=None)
    servicer._scene = scene
    servicer._initialized = True

    resp = servicer.SetLod(pb2.SetLodRequest(enabled=True), _CTX)
    assert resp.success is True
    assert resp.enabled is False
    assert scene.lod_enabled is False
    assert "unavailable" in resp.message.lower()
