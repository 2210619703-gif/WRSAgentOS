"""A detector ID must never be accepted as a segmentation ID, even though both are strings."""

import json

import pytest
from conftest import eventually
from test_runtime import OfflineNode

from wrs_agent.executor import ActionExecutor
from wrs_agent.nodes.grasp.node import GraspState
from wrs_agent.planner.providers import ModelRequest
from wrs_agent.planner.providers.wire import decision_schema
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import Plan, Step, TaskRequest, new_id
from wrs_agent.skills.vision.grasping import (
    INFER_GRASPS,
    LOCATE,
    SEGMENT,
    SEGMENT_REGION,
    GraspsOutput,
    LocationOutput,
    SegmentOutput,
)


@pytest.fixture
async def pipeline(tmp_path):
    async def locate(*_):
        return LocationOutput(location_id="location", observation_id="observation")

    async def segment(*_):
        return SegmentOutput(segment_id="segment", observation_id="observation")

    async def infer(*_):
        return GraspsOutput(candidate_set_id="grasps", observation_id="observation", count=1)

    executors = {}
    bindings = {}
    for node, skills in {
        "yolo": (LOCATE.bind(locate),),
        "segmentation": (SEGMENT_REGION.bind(segment), SEGMENT.bind(segment)),
        "graspnet": (INFER_GRASPS.bind(infer),),
    }.items():
        executor = ActionExecutor(tmp_path / f"{node}.sqlite3", state=GraspState(),
                                  skills=skills, backend="fixture",
                                  features_extra={"robot_controls": False})
        executor.node_id = node
        executors[node] = executor
        bindings.update({skill.name: node for skill in skills})
    runtime = Runtime({node: OfflineNode(ex) for node, ex in executors.items()}, bindings)
    try:
        yield runtime, executors
    finally:
        await runtime.close()
        for executor in executors.values():
            await executor.close()


@pytest.mark.parametrize("field", ["location_id", "observation_id"])
async def test_locate_cannot_skip_segmentation_before_any_action(pipeline, field):
    runtime, executors = pipeline
    plan = Plan(steps=[
        Step(step_id="locate_cup", skill="locate_object", args={"target": "cup"}),
        Step(step_id="infer_grasp", skill="infer_grasps", depends_on=["locate_cup"],
             inputs={"segment_id": {"step_id": "locate_cup", "field": field}}),
    ])
    await runtime.start(TaskRequest(request_id=new_id(), plan=plan))
    result = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
    assert result["error"]["code"] == "artifact_type_mismatch"
    assert result["error"]["stage"] == "preflight"
    assert result["error"]["node_id"] == "graspnet"
    assert "segment_id" in result["error"]["message"]
    assert all(ex.executions == 0 and not ex.records for ex in executors.values())


@pytest.mark.parametrize("region", [True, False])
async def test_both_real_skill_contract_chains_bind_correct_results(pipeline, region):
    runtime, executors = pipeline
    steps = [Step(step_id="segment", skill="segment_object", args={"target": "cup"})]
    if region:
        steps = [
            Step(step_id="locate", skill="locate_object", args={"target": "cup"}),
            Step(step_id="segment", skill="segment_region", depends_on=["locate"],
                 inputs={"location_id": {"step_id": "locate", "field": "location_id"}}),
        ]
    steps.append(Step(
        step_id="infer", skill="infer_grasps", depends_on=["segment"],
        inputs={"segment_id": {"step_id": "segment", "field": "segment_id"}},
    ))
    await runtime.start(TaskRequest(request_id=new_id(), plan=Plan(steps=steps)))
    result = await eventually(runtime.snapshot, lambda s: s["state"] in {"SUCCEEDED", "FAILED"})
    assert result["state"] == "SUCCEEDED", result
    request, status = next(iter(executors["graspnet"].records.values()))
    assert request["args"] == {"segment_id": "segment"}
    assert status.output["candidate_set_id"] == "grasps"
    assert executors["yolo"].executions == int(region)


def test_artifact_annotations_stay_local_to_contract_validation():
    skills = [skill.spec for skill in (LOCATE, SEGMENT_REGION, INFER_GRASPS)]
    assert skills[0].output["properties"]["location_id"]["x-wrs-artifact"] == "location"
    assert skills[1].parameters["properties"]["location_id"]["x-wrs-artifact"] == "location"
    assert skills[1].output["properties"]["segment_id"]["x-wrs-artifact"] == "segment"
    assert skills[2].parameters["properties"]["segment_id"]["x-wrs-artifact"] == "segment"
    request = ModelRequest(goal="predict grasps", context={
        "skills": [skill.model_dump() for skill in skills],
    })
    schema = decision_schema(request)
    assert "x-wrs-artifact" not in json.dumps(schema)
    # Schema generation must not erase the annotations the Runtime needs.
    assert INFER_GRASPS.spec.parameters == skills[2].parameters
