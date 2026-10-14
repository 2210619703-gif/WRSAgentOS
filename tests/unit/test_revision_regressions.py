import asyncio

import pytest
from conftest import action, eventually
from test_runtime import OfflineNode, motion

from wrs_agent.bindings import load_bindings
from wrs_agent.nodes.tts.backend import make_mock_tts
from wrs_agent.nodes.voice import VoiceNode
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import ActionStatus, Plan, Step, TaskCancelRequest, TaskRequest


@pytest.mark.parametrize("change", ["restart", "epoch"])
async def test_entire_plan_binds_before_first_branch(make_env, tmp_path, change):
    old, new = make_env(), make_env()
    speech = make_mock_tts(tmp_path / "speech.db", duration=0.12)
    robot = OfflineNode(old)
    runtime = Runtime({"wrs": robot, "tts": OfflineNode(speech)}, load_bindings()[1])
    plan = Plan(
        steps=[
            Step(step_id="say", skill="speak", args={"text": "wait"}),
            Step(step_id="move", skill="move_named_pose", args={"pose": "B"}, depends_on=["say"]),
            Step(step_id="dependent", skill="speak", args={"text": "never"}, depends_on=["move"]),
        ]
    )
    try:
        await runtime.start(TaskRequest(request_id="start", plan=plan))
        await eventually(speech.snapshot, lambda s: s.active_action is not None)
        if change == "restart":
            robot.executor = new
        else:
            old.epoch += 1
        result = await eventually(runtime.snapshot, lambda s: s["state"] != "RUNNING")
        assert result["state"] == "CANCELLED"
        assert result["steps"] == {"say": "SUCCEEDED", "move": "STALE", "dependent": "BLOCKED"}
        assert old.executions == new.executions == 0
        assert speech.executions == 1
    finally:
        await runtime.close()
        await old.close()
        await new.close()
        await speech.close()


async def test_cancel_then_start_ignore_unrelated_offline_node(make_env):
    env = make_env(duration=0.1)

    class Offline:
        async def features(self):
            raise AssertionError("unrelated node queried")

        async def snapshot(self, **kwargs):
            raise AssertionError("unrelated node queried")

    runtime = Runtime({"wrs": OfflineNode(env), "tts": Offline()}, load_bindings()[1])
    try:
        first = await runtime.start(TaskRequest(request_id="start", plan=motion()))
        await eventually(env.snapshot, lambda s: s.active_action is not None)
        receipt = await runtime.cancel(
            TaskCancelRequest(request_id="hold", task_id=first["task_id"])
        )
        assert receipt["accepted"]
        await eventually(runtime.snapshot, lambda s: s["state"] == "CANCELLED")
        await runtime.start(TaskRequest(request_id="next", plan=motion("C")))
        await eventually(runtime.snapshot, lambda s: s["state"] == "SUCCEEDED")
        assert env.world.pose == "C"
    finally:
        await runtime.close()
        await env.close()


@pytest.mark.parametrize("historical", ["SUCCEEDED", "RUNNING"])
async def test_tts_restart_retains_dedup_and_only_reopens_known_history(tmp_path, historical):
    path = tmp_path / "tts.db"
    tts = make_mock_tts(path, duration=0.01)
    request = action(tts, "speak", {"text": "old"})
    await tts.journal.save(
        request.model_dump(),
        ActionStatus(
            action_id=request.action_id,
            state=historical,
            verification="PASS" if historical == "SUCCEEDED" else "PENDING",
        ),
    )
    await tts.close()
    restarted = make_mock_tts(path, duration=0.01)
    try:
        assert restarted.executions == 0
        expected = "SUCCEEDED" if historical == "SUCCEEDED" else "UNKNOWN"
        assert restarted.status(request.action_id).state == expected
        assert (await restarted.submit(request)).status.state == expected
        receipt = await restarted.submit(action(restarted, "speak", {"text": "new"}))
        assert receipt.accepted == (historical == "SUCCEEDED")
        if receipt.accepted:
            await restarted.runner
            assert restarted.executions == 1
        else:
            assert restarted.admission == "UNKNOWN"
    finally:
        await restarted.close()


@pytest.mark.parametrize("failure", ["snapshot", "lost_reply"])
async def test_voice_retry_and_concurrent_duplicates_share_control(make_env, failure):
    env = make_env()
    node = OfflineNode(env)
    calls, snapshots = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def snapshot(**kwargs):
        snapshots.append(True)
        if failure == "snapshot" and len(snapshots) == 1:
            raise TimeoutError("snapshot unavailable")
        return env.snapshot()

    async def control(kind, request):
        calls.append(request)
        receipt = await env.control(kind, request)
        if failure == "lost_reply" and len(calls) == 1:
            raise TimeoutError("lost reply after effect")
        entered.set()
        await release.wait()
        return receipt

    class Bus:
        def __init__(self):
            self.handlers = {}

        def register_handler(self, key, handler, **kwargs):
            self.handlers[key] = handler

        def publish(self, *args):
            pass

    node.snapshot, node.control = snapshot, control
    bus = Bus()
    voice = VoiceNode()
    voice.transport, voice.wrs = bus, node
    handle = voice.control_event
    event = {"event_id": "stop-once", "kind": "stop"}
    try:
        with pytest.raises(TimeoutError):
            await handle(event)
        first = asyncio.create_task(handle(event))
        await entered.wait()
        second = asyncio.create_task(handle(event))
        await asyncio.sleep(0)
        release.set()
        a, b = await asyncio.gather(first, second)
        assert a == b and a["accepted"] and a["phase"] == "STOPPED"
        assert env.epoch == 1
        assert len(calls) == (2 if failure == "lost_reply" else 1)
        assert all(c == calls[0] for c in calls)
        assert await handle(event) == a
        assert len(snapshots) == (2 if failure == "snapshot" else 1)
        with pytest.raises(ValueError, match="event_id_conflict"):
            await handle({**event, "kind": "barge_in"})
    finally:
        release.set()
        await voice.teardown()
        await env.close()


async def test_task_history_survives_queue_cancel_and_returns_detached_results(make_env):
    env = make_env(duration=0.08)
    runtime = Runtime({"wrs": OfflineNode(env)}, load_bindings()[1])
    try:
        first = await runtime.start(TaskRequest(request_id="one", plan=motion()))
        queued = await runtime.enqueue(TaskRequest(request_id="two", plan=motion("C")))
        assert runtime.task_status(queued["task_id"])["state"] == "QUEUED"
        await eventually(
            runtime.snapshot,
            lambda s: s["task_id"] == queued["task_id"] and s["state"] == "SUCCEEDED",
        )
        assert runtime.task_status(first["task_id"])["state"] == "SUCCEEDED"
        snapshot = runtime.task_status(first["task_id"])
        snapshot["steps"].clear()
        assert runtime.task_status(first["task_id"])["steps"] == {"move": "SUCCEEDED"}
        third = await runtime.start(TaskRequest(request_id="three", plan=motion()))
        discarded = await runtime.enqueue(TaskRequest(request_id="four", plan=motion("C")))
        await runtime.cancel(TaskCancelRequest(request_id="hold", task_id=third["task_id"]))
        await eventually(runtime.snapshot, lambda s: s["state"] == "CANCELLED")
        assert runtime.task_status(third["task_id"])["state"] == "CANCELLED"
        assert runtime.task_status(discarded["task_id"])["state"] == "CANCELLED"
        newer = await runtime.start(TaskRequest(request_id="next", plan=motion("C")))
        await eventually(runtime.snapshot, lambda s: s["state"] == "SUCCEEDED")
        assert runtime.task_status(third["task_id"])["state"] == "CANCELLED"
        assert newer["task_id"] != third["task_id"]
        fresh_runtime = Runtime({}, {})
        with pytest.raises(ValueError, match="task_not_found"):
            fresh_runtime.task_status(first["task_id"])
    finally:
        await runtime.close()
        await env.close()


async def test_result_capacity_rejects_without_overwriting_existing_records(make_env, make_llm):
    from wrs_agent.planner import ModelPlanner
    from wrs_agent.schemas import GoalRequest

    env = make_env(duration=0.01)
    runtime = Runtime(
        {"wrs": OfflineNode(env)},
        load_bindings()[1],
        ModelPlanner(make_llm('{"kind":"answer","text":"saved answer"}')),
    )
    try:
        await runtime.goal(GoalRequest(request_id="question", goal="status"))
        await runtime.planning.worker
        assert runtime.goal_status("question")["state"] == "ANSWER"
        first = await runtime.start(TaskRequest(request_id="first", plan=motion()))
        await eventually(runtime.snapshot, lambda s: s["state"] == "SUCCEEDED")
        assert runtime.goal_status("question")["reason"] == "saved answer"
        runtime.goals.update({f"filled-{i}": {} for i in range(4094)})
        with pytest.raises(ValueError, match="result_capacity"):
            await runtime.start(TaskRequest(request_id="overflow", plan=motion()))
        with pytest.raises(ValueError, match="result_capacity"):
            await runtime.goal(GoalRequest(request_id="overflow-goal", goal="status"))
        assert runtime.task_status(first["task_id"])["state"] == "SUCCEEDED"
        assert "overflow" not in runtime.requests and "overflow-goal" not in runtime.goals
    finally:
        await runtime.close()
        await env.close()


async def test_control_epoch_and_business_state_version_are_independent(make_env):
    from conftest import control

    env = make_env(duration=0.01)
    try:
        old = action(env)
        version = env.world.version
        await env.hold(control(env))
        assert env.world.version == version
        assert (await env.submit(old)).reason == "stale_epoch"
        await env.allow_actions(control(env, state_version=version))
        stale_state = action(env, "observe", {})
        epoch = env.epoch
        await env.submit(action(env))
        await env.runner
        assert env.epoch == epoch and env.world.version > version
        assert (await env.submit(stale_state)).reason == "stale_state"
    finally:
        await env.close()


async def test_submitted_action_lost_on_restart_is_unknown_but_other_branch_finishes(
    make_env, tmp_path
):
    old, new = make_env(duration=0.3), make_env()
    tts = make_mock_tts(tmp_path / "independent.db", duration=0.12)
    node = OfflineNode(old)
    runtime = Runtime({"wrs": node, "tts": OfflineNode(tts)}, load_bindings()[1])
    try:
        plan = Plan(
            steps=[
                Step(step_id="move", skill="move_named_pose", args={"pose": "B"}),
                Step(step_id="say", skill="speak", args={"text": "independent"}),
                Step(
                    step_id="after",
                    skill="move_named_pose",
                    args={"pose": "C"},
                    depends_on=["move"],
                ),
            ]
        )
        await runtime.start(TaskRequest(request_id="start", plan=plan))
        await eventually(old.snapshot, lambda s: s.active_action is not None)
        node.executor = new
        result = await eventually(runtime.snapshot, lambda s: s["state"] == "UNKNOWN")
        assert result["steps"]["move"] == "UNKNOWN"
        assert result["steps"]["after"] == "BLOCKED"
        assert result["steps"]["say"] == "SUCCEEDED"
        assert new.executions == 0 and new.epoch == 0
    finally:
        await runtime.close()
        await old.close()
        await new.close()
        await tts.close()


async def test_cancelling_blocks_new_tasks_until_all_old_resources_stop(make_env, tmp_path):
    env = make_env(duration=0.1)
    tts = make_mock_tts(tmp_path / "cancel.db", duration=0.01)
    runtime = Runtime({"wrs": OfflineNode(env), "tts": OfflineNode(tts)}, load_bindings()[1])
    waiting, release = asyncio.Event(), asyncio.Event()
    wait_stopped = runtime._wait_stopped

    async def delayed(name, previous=None):
        if name == "wrs":
            waiting.set()
            await release.wait()
        return await wait_stopped(name, previous)

    runtime._wait_stopped = delayed
    speech = Plan(steps=[Step(step_id="say", skill="speak", args={"text": "new"})])
    try:
        first = await runtime.start(TaskRequest(request_id="one", plan=motion()))
        await eventually(env.snapshot, lambda s: s.active_action is not None)
        request = TaskCancelRequest(request_id="cancel", task_id=first["task_id"])
        receipts = await asyncio.gather(runtime.cancel(request), runtime.cancel(request))
        assert receipts[0] == receipts[1] and receipts[0]["accepted"]
        await waiting.wait()
        for index in range(2):
            with pytest.raises(ValueError, match="task_busy"):
                await runtime.start(TaskRequest(request_id=f"next-{index}", plan=speech))
        assert tts.executions == 0 and runtime.execution.state == "CANCELLING"
        release.set()
        await eventually(runtime.snapshot, lambda s: s["state"] == "CANCELLED")
        assert runtime.task_status(first["task_id"])["state"] == "CANCELLED"
        second = await runtime.start(TaskRequest(request_id="ready", plan=speech))
        await eventually(runtime.snapshot, lambda s: s["state"] == "SUCCEEDED")
        assert runtime.task_id == second["task_id"] and tts.executions == 1
    finally:
        release.set()
        await runtime.close()
        await env.close()
        await tts.close()


async def test_control_change_waits_for_known_action_stop_result(make_env):
    from conftest import control

    env = make_env(duration=0.3)
    node = OfflineNode(env)
    original = node.status
    lagged = 0

    async def lagged_status(action_id):
        nonlocal lagged
        # A normal status reply can lag a control reply. CANCELLING is not UNKNOWN.
        if env.epoch and lagged < 2:
            lagged += 1
            return ActionStatus(action_id=action_id, state="CANCELLING")
        return await original(action_id)

    node.status = lagged_status
    runtime = Runtime({"wrs": node}, load_bindings()[1])
    try:
        await runtime.start(TaskRequest(request_id="start", plan=motion()))
        await eventually(env.snapshot, lambda s: s.active_action is not None)
        assert (await env.hold(control(env))).accepted
        result = await eventually(runtime.snapshot, lambda s: s["state"] != "RUNNING")
        assert lagged == 2 and result["state"] == "CANCELLED"
        assert result["steps"] == {"move": "CANCELLED"}
    finally:
        await runtime.close()
        await env.close()


async def test_later_planning_failure_does_not_change_completed_task(make_env, make_llm):
    from wrs_agent.planner import ModelPlanner
    from wrs_agent.schemas import GoalRequest

    env = make_env(duration=0.01)
    runtime = Runtime(
        {"wrs": OfflineNode(env)},
        load_bindings()[1],
        ModelPlanner(
            make_llm(
                '{"kind":"execute","plan":{"steps":[{"step_id":"bad","skill":"unregistered"}]}}'
            )
        ),
    )
    try:
        first = await runtime.start(TaskRequest(request_id="first", plan=motion()))
        await eventually(runtime.snapshot, lambda s: s["state"] == "SUCCEEDED")
        await runtime.goal(GoalRequest(request_id="bad-goal", goal="unsupported"))
        await runtime.planning.worker
        assert runtime.goal_status("bad-goal")["state"] == "FAILED"
        assert runtime.task_status(first["task_id"])["state"] == "SUCCEEDED"
        assert runtime.task_status(first["task_id"])["reason"] == ""
        snapshot = runtime.snapshot()
        assert snapshot["task_id"] == first["task_id"]
        assert snapshot["state"] == "SUCCEEDED" and snapshot["reason"] == ""
        assert snapshot["error"] is None
        assert snapshot["planning"] == "FAILED"
        assert snapshot["planning_reason"] == runtime.goal_status("bad-goal")["reason"]
        assert snapshot["planning_error"] == runtime.goal_status("bad-goal")["error"]

        runtime.planner = ModelPlanner(make_llm('{"kind":"answer","text":"你好"}'))
        await runtime.goal(GoalRequest(request_id="answer", goal="你好"))
        await runtime.planning.worker
        snapshot = runtime.snapshot()
        assert snapshot["state"] == "SUCCEEDED" and snapshot["reason"] == ""
        assert snapshot["planning_reason"] == "你好" and snapshot["planning_error"] is None
    finally:
        await runtime.close()
        await env.close()
