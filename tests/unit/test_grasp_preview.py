"""Same-observation target coloring and replacement races; no devices or model calls."""

import importlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tests.fixtures.grasping.virtual_data import data
from wrs_agent.errors import AgentError
from wrs_agent.nodes.camera import preview


def test_grasp_cloud_colors_the_mask_and_keeps_background_and_input_arrays():
    grasp = data()
    segment = grasp.segment
    original = {key: value.copy() for key, value in segment.arrays.items()}
    value = preview.make_grasp_preview(grasp)
    assert value["observation_id"] == segment.info.observation.observation_id
    assert value["candidate_set_id"] == grasp.info.candidate_set_id
    assert value["segment_id"] == segment.info.segment_id
    assert len(value["points"]) == 768 and value["segmented_point_count"] == 64
    target = np.asarray(value["points"])[:, 2] < .4
    colors = np.asarray(value["colors"])
    assert target.sum() == 64
    assert np.all(colors[target] == preview.SEGMENT_POINT_COLOR)
    assert np.all(colors[~target] == [0, 0, 0])  # Original background RGB.
    assert value["grasps"][0] == grasp.info.candidates[0].model_dump()
    for key, array in original.items():
        np.testing.assert_array_equal(segment.arrays[key], array)


@pytest.mark.parametrize("mask_kind", ["one_pixel", "whole_frame", "invalid_target"])
def test_cloud_budget_preserves_small_targets_and_rejects_invalid_points(monkeypatch, mask_kind):
    grasp = data()
    arrays = {key: value.copy() for key, value in grasp.segment.arrays.items()}
    arrays["points"][0, :3] = [[np.nan, 0, .3], [0, 0, 0], [0, 0, -.1]]
    mask = np.zeros((24, 32), dtype=bool)
    mask[1, 1] = True  # Off the normal preview sampling grid.
    if mask_kind == "whole_frame":
        mask[:] = True
    elif mask_kind == "invalid_target":
        mask[:] = False
        mask[0, :3] = True
    segment = replace(grasp.segment, arrays=arrays, mask=mask,
                      info=grasp.segment.info.model_copy(update={"pixel_count": int(mask.sum())}))
    monkeypatch.setattr(preview, "MAX_PREVIEW_POINTS", 8)
    value = preview.make_grasp_preview(replace(grasp, segment=segment))
    points = np.asarray(value["points"])
    assert len(points) == 8 and len(np.unique(points, axis=0)) == 8
    assert np.isfinite(points).all() and np.all(points[:, 2] > 0)
    expected = {"one_pixel": 1, "whole_frame": 8, "invalid_target": 0}[mask_kind]
    assert value["segmented_point_count"] == expected
    if mask_kind == "one_pixel":
        np.testing.assert_allclose(points[-1], arrays["points"][1, 1], atol=1e-5)
    assert value == preview.make_grasp_preview(replace(grasp, segment=segment))


@pytest.mark.parametrize("change", ["cancelled", "candidate", "task", "observation"])
async def test_grasp_preview_drops_previous_data_and_rejects_late_replaced_result(
    monkeypatch, change,
):
    grasp = data()
    ui = preview.CameraPreview(endpoint="unused", env_id="test")
    ui.latest["grasps"] = {"state": "ready", "candidate_set_id": "old", "points": [[0, 0, 1]]}
    ref = dict(candidate_set_id="candidate", task_id="task", observation_id="fixture.1", age_s=0)
    changed = AgentError("artifact_invalidated") if change == "cancelled" else ref | {
        {"candidate": "candidate_set_id", "task": "task_id", "observation": "observation_id"}
        [change]: "replacement",
    }
    ui.grasp_bus = SimpleNamespace(request=AsyncMock(side_effect=[ref, changed]))

    async def read(*_):
        assert "grasps" not in ui.latest  # Release old data before awaiting a new packet.
        return grasp, 0

    monkeypatch.setattr(preview, "read_grasps", read)
    await ui._update("grasps")
    assert ui.latest["grasps"]["state"] == "unavailable"
    assert "points" not in ui.latest["grasps"] and "grasps" not in ui.latest["grasps"]
    assert ui.latest["grasps"]["reason"] == (
        "artifact_invalidated" if change == "cancelled" else "grasp_changed"
    )


def test_viewer_shows_generated_count_from_task_output_without_claiming_live_availability():
    describe = importlib.import_module("examples.voice.09_viewer").describe_results
    overview = {"steps": {"infer_spoon": "SUCCEEDED"}, "step_outputs": {
        "infer_spoon": {"candidate_set_id": "id", "observation_id": "frame", "count": 1},
    }}
    assert describe(overview) == "infer_spoon：已生成 1 个抓取候选"
    # No new step may inherit the previous task's count.
    assert describe({"steps": {}, "step_outputs": {}}) == "（无）"
    overview["steps"]["infer_spoon"] = "FAILED"
    assert describe(overview) == "（无）"
    assert "重启 Agent" in describe({"steps": {}})


@pytest.mark.parametrize("reason", ["artifact_expired", "artifact_invalidated", "request_timeout"])
async def test_grasp_query_failure_keeps_specific_cause_and_drops_geometry(reason):
    ui = preview.CameraPreview(endpoint="unused", env_id="test")
    ui.latest["grasps"] = {"state": "ready", "candidate_set_id": "old", "points": [[0, 0, 1]]}
    ui.grasp_bus = SimpleNamespace(request=AsyncMock(side_effect=AgentError(reason)))
    await ui._update("grasps")
    assert ui.latest["grasps"] == {"state": "unavailable", "reason": reason}


def test_grasp_http_shows_time_remaining_then_withholds_expired_geometry(monkeypatch):
    # The unit environment excludes WRS/websockets; stub only its HTTP response boundary.
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "websockets.datastructures", SimpleNamespace(Headers=dict))
    monkeypatch.setitem(sys.modules, "websockets.http11", SimpleNamespace(
        Response=lambda status, reason, headers, body: SimpleNamespace(body=body),
    ))
    now = [100.]
    monkeypatch.setattr(preview, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ui = preview.CameraPreview(endpoint="unused", env_id="test")
    ui.latest["grasps"] = preview.make_grasp_preview(data())
    ui.observed_at["grasps"], ui.checked_at["grasps"] = 71., 100.
    ready = json.loads(ui.http_response("/camera/preview?view=grasps", None).body)
    assert ready["grasp_count"] == 1 and ready["valid_for_s"] == 1
    now[0], ui.checked_at["grasps"] = 102., 102.
    stale = json.loads(ui.http_response("/camera/preview?view=grasps", None).body)
    assert stale["state"] == "stale" and "points" not in stale and "grasps" not in stale
