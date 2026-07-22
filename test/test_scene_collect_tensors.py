from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from splatsim.background import Background
from splatsim.rigid_body import RigidBody
from splatsim.scene import Scene


def _make_scene() -> tuple[Scene, object, object]:
    """Build a LOD-disabled Scene with stub background and rigid body.

    ``collect_tensors`` only touches ``.tensors`` and ``.lod_index`` when LOD
    is disabled (``lod_manager=None``), so lightweight stubs suffice and no GPU
    tensors are required.
    """
    bg_tensors = object()
    rb_tensors = object()
    background = cast(Background, SimpleNamespace(tensors=bg_tensors, lod_index=None))
    rigid_body = cast(RigidBody, SimpleNamespace(tensors=rb_tensors, lod_index=None))
    scene = Scene(background=background, rigid_bodies={"car": rigid_body})
    return scene, bg_tensors, rb_tensors


def test_collect_tensors_includes_background_by_default() -> None:
    scene, bg_tensors, rb_tensors = _make_scene()

    result = scene.collect_tensors()

    assert bg_tensors in result
    assert rb_tensors in result


def test_collect_tensors_omits_background_when_disabled() -> None:
    scene, bg_tensors, rb_tensors = _make_scene()

    result = scene.collect_tensors(include_background=False)

    assert bg_tensors not in result
    assert rb_tensors in result
