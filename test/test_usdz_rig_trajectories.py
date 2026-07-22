from __future__ import annotations

import json
import zipfile

from splatsim._usdz import read_rig_trajectories


def test_read_rig_trajectories_accepts_alpasim_layout(tmp_path) -> None:
    usdz_path = tmp_path / "scene.usdz"
    rig_uri = "rig_trajectories.json"
    rig_to_world = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    world_to_nre = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rig_doc = {
        "world_to_nre": {"matrix": world_to_nre},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [rig_to_world],
                "T_rig_world_timestamps_us": [123],
            }
        ],
        "camera_calibrations": {},
        "lidar_calibrations": {},
    }

    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr(rig_uri, json.dumps(rig_doc))

    rigs = read_rig_trajectories(usdz_path, rig_uri)

    assert len(rigs) == 1
    assert rigs[0].rig_id == "ego"
    assert len(rigs[0].poses) == 1
    assert rigs[0].poses[0].timestamp_us == 123
    assert rigs[0].poses[0].translation == (11.0, 22.0, 33.0)
