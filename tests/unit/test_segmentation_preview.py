"""Selected-mask previews and public failures use the real node/Action boundaries."""

import asyncio
import base64
import io
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from conftest import eventually
from PIL import Image
from test_runtime import OfflineNode

from tests.fixtures.grasping.virtual_data import data
from wrs_agent.errors import AgentError
from wrs_agent.executor import ActionExecutor
from wrs_agent.nodes.camera.preview import CameraPreview, make_segment_preview
from wrs_agent.nodes.segmentation.data import LatestSegment
from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.nodes.segmentation.node import SegmentationNode, SegmentationState
from wrs_agent.nodes.vision.artifacts import ArtifactCache, ArtifactRequest
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import ActionState, Plan, Step, TaskRequest, new_id
from wrs_agent.skills.vision.grasping import SEGMENT_REGION


@pytest.fixture
def segment():
    return data().segment


def test_overlay_uses_retained_frame_without_mutating_rgb_or_mask(segment):
    rgb, mask = segment.arrays["rgb"].copy(), segment.mask.copy()
    value = make_segment_preview(segment)
    assert value["observation_id"] == segment.info.observation.observation_id
    assert value["segment_id"] == segment.info.segment_id
    assert value["pixel_count"] == int(mask.sum())
    encoded = value["segmentation"].split(",")[1]
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
    assert image[12, 16, 1] > rgb[12, 16, 1]  # Green mask, not just the original image.
    np.testing.assert_array_equal(segment.arrays["rgb"], rgb)
    np.testing.assert_array_equal(segment.mask, mask)
    assert "points" not in value and "depth" not in value


async def test_latest_requires_successful_current_unexpired_artifact(segment):
    node = object.__new__(SegmentationNode)
    node.latest_request = None
    with pytest.raises(AgentError, match="segment_not_available"):
        await node.latest({})
    node.cache = ArtifactCache()
    state = SimpleNamespace(state=ActionState.RUNNING)
    node.executor = SimpleNamespace(boot_id="boot", epoch=0, status=lambda _: state)
    node.cache.put("segment", segment.info.owner, segment, nbytes=segment.nbytes,
                   observed_at=time.monotonic() - 5)
    node.latest_request = ArtifactRequest(artifact_id="segment", task_id="task")
    with pytest.raises(AgentError, match="artifact_invalidated"):
        await node.latest({})
    state.state = ActionState.SUCCEEDED
    ref = LatestSegment.model_validate(await node.latest({}))
    assert ref.segment_id == "segment" and ref.task_id == "task" and ref.age_s >= 5
    node.executor.epoch += 1
    with pytest.raises(AgentError, match="artifact_invalidated"):
        await node.latest({})
    node.cache.ttl = 1
    with pytest.raises(AgentError, match="artifact_expired"):
        await node.latest({})


async def test_preview_queries_mask_once_checks_each_poll_and_never_reads_latest_camera(
    segment, monkeypatch,
):
    from wrs_agent.nodes.camera import preview

    ui = CameraPreview(endpoint="unused", env_id="test")
    ref = LatestSegment(segment_id="segment", task_id="task", observation_id="fixture.1", age_s=5)
    ui.segmentation_bus = SimpleNamespace(request=AsyncMock(return_value=ref.model_dump()))
    ui.client = SimpleNamespace(read=AsyncMock(), status=AsyncMock())
    read = AsyncMock(return_value=(segment, 5))
    monkeypatch.setattr(preview, "read_segment", read)
    await ui._update("segmentation")
    assert ui.latest["segmentation"]["state"] == "ready"
    assert time.monotonic() - ui.observed_at["segmentation"] >= 5
    await ui._update("segmentation")
    read.assert_awaited_once_with(ui.segmentation_bus, "segment", "task")
    ui.client.read.assert_not_awaited()
    ui.segmentation_bus.request.side_effect = AgentError("artifact_invalidated")
    await ui._update("segmentation")
    assert ui.latest["segmentation"] == {"state": "unavailable", "reason": "artifact_invalidated"}
    ui.client.status.assert_not_awaited()


@pytest.mark.parametrize("change", ["observation", "cancel", "replaced"])
async def test_preview_does_not_publish_mismatched_or_revoked_result(segment, monkeypatch, change):
    from wrs_agent.nodes.camera import preview

    ui = CameraPreview(endpoint="unused", env_id="test")
    ref = dict(segment_id="segment", task_id="task", observation_id="fixture.1", age_s=0)
    replies = [ref, AgentError("artifact_invalidated") if change == "cancel"
               else ref | {"segment_id": "replacement"}]
    if change == "observation":
        replies[0] = ref | {"observation_id": "different"}
    ui.segmentation_bus = SimpleNamespace(request=AsyncMock(side_effect=replies))
    monkeypatch.setattr(preview, "read_segment", AsyncMock(return_value=(segment, 0)))
    await ui._update("segmentation")
    assert ui.latest["segmentation"]["state"] != "ready"
    assert "segmentation" not in ui.latest["segmentation"]


@pytest.mark.parametrize("where,failure,code,node_id,stage", [
    ("connect", AgentError("request_timeout", node_id="yolo", stage="discovery"),
     "segmentation_source_timeout", "yolo", "discovery"),
    ("read", TimeoutError("private details"), "segmentation_source_timeout", "yolo", "observe"),
    ("read", AgentError("artifact_task_mismatch"),
     "artifact_task_mismatch", "segmentation", "observe"),
    ("infer", ValueError("region_mask_not_found"),
     "region_mask_not_found", "segmentation", "observe"),
    ("infer", ValueError("region_mask_ambiguous"),
     "region_mask_ambiguous", "segmentation", "observe"),
    ("infer", ValueError("private details"),
     "invalid_segmentation_output", "segmentation", "observe"),
    ("infer", RuntimeError("private details"),
     "segmentation_inference_failed", "segmentation", "observe"),
])
async def test_segmentation_failure_survives_action_runtime_without_private_details(
    tmp_path, monkeypatch, segment, where, failure, code, node_id, stage,
):
    from wrs_agent.nodes.segmentation import node as segmentation_module

    node = object.__new__(SegmentationNode)
    node.options = SimpleNamespace(backend="fastsam-box", locator="yolo")
    node.registry = SimpleNamespace(refresh=AsyncMock())
    node.connect = Mock(side_effect=failure if where == "connect" else None)
    node.loop, node.worker, node.cache = asyncio.get_running_loop(), None, ArtifactCache()
    node.segmenter = SimpleNamespace(segment_box=Mock(side_effect=failure))
    location = SimpleNamespace(info=SimpleNamespace(observation=segment.info.observation,
                               target="cup", object=SimpleNamespace(box=[1, 1, 20, 20])),
                               arrays=segment.arrays)
    monkeypatch.setattr(segmentation_module, "read_location", AsyncMock(
        return_value=(location, 0), side_effect=failure if where == "read" else None,
    ))
    executor = ActionExecutor(tmp_path / "sam.sqlite3", state=SegmentationState(),
                              skills=(SEGMENT_REGION.bind(node.segment),), backend="fixture",
                              features_extra={"robot_controls": False})
    node.executor, executor.node_id = executor, "segmentation"
    runtime = Runtime({"segmentation": OfflineNode(executor)}, {"segment_region": "segmentation"})
    try:
        await runtime.start(TaskRequest(request_id=new_id(), plan=Plan(steps=[
            Step(step_id="segment", skill="segment_region", args={"location_id": "location"}),
        ])))
        result = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
        assert result["error"]["code"] == code
        assert result["error"]["node_id"] == node_id and result["error"]["stage"] == stage
        assert "private details" not in str(result)
        assert node.latest_request is None
        assert node.segmenter.segment_box.call_count == int(where == "infer")
    finally:
        await runtime.close()
        await executor.close()


def test_fastsam_oom_is_classified_before_any_mask_is_returned(segment):
    class OutOfMemoryError(RuntimeError):
        pass

    model = object.__new__(FastSamSegmenter)
    model.torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
    model.device, model.confidence = "cuda", .4
    model.model = SimpleNamespace(predict=Mock(side_effect=OutOfMemoryError("private details")))
    with pytest.raises(ValueError, match="^segmentation_gpu_out_of_memory$"):
        model.masks(segment.arrays["rgb"])
