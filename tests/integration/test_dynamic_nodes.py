"""Nodes join a real Zenoh environment without a shared deployment file."""

import asyncio
import json
import socket
import subprocess
from pathlib import Path

import pytest
from conftest import eventually, submit_request

from wrs_agent import AgentError, System, step
from wrs_agent.executor import ActionExecutor
from wrs_agent.nodes import Node
from wrs_agent.nodes.agent import AgentNode
from wrs_agent.nodes.tts.backend import SpeechState
from wrs_agent.processes import NO_WINDOW, check_router_version, router_path
from wrs_agent.registry import NodeRegistry, register_node
from wrs_agent.schemas import ActionRequest, Empty, new_id
from wrs_agent.skills import Skill
from wrs_agent.transport import Transport

pytestmark = pytest.mark.zenoh


def skill(name, handler):
    return Skill(
        name=name,
        description="Count one completed test action",
        resources=["counter"],
        preconditions=[],
        verification="counter_state",
        arguments=Empty,
    ).bind(handler)


def count(state, args, stop, progress):
    state.completed += 1
    return True


class CounterNode(Node):
    action_service = True

    async def setup(self):
        self.actions(ActionExecutor(
            self.journal, state=SpeechState(), backend="test",
            skills=[skill("count", count)],
            features_extra={"robot_controls": False, "controller_flush": False},
        ))


class GateNode(Node):
    action_service = True

    async def setup(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def hold(state, args, stop, progress):
            self.entered.set()
            while not self.release.is_set():
                if stop.is_set():
                    return False
                try:
                    await asyncio.wait_for(self.release.wait(), 0.02)
                except TimeoutError:
                    pass
            return count(state, args, stop, progress)

        self.actions(ActionExecutor(
            self.journal, state=SpeechState(), backend="test",
            skills=[skill("gate", hold)],
            features_extra={"robot_controls": False, "controller_flush": False},
        ))


@pytest.fixture
async def network(tmp_path, monkeypatch):
    # Start only a router. No deployment file or built-in worker seeds discovery.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    config = tmp_path / "router.json5"
    config.write_text(json.dumps({
        "mode": "router", "listen": {"endpoints": [endpoint]},
        "scouting": {"multicast": {"enabled": False}, "gossip": {"enabled": False}},
        "adminspace": {"enabled": False}, "plugins_loading": {"enabled": False},
    }), encoding="utf-8")
    router = router_path()
    await asyncio.to_thread(check_router_version, router)
    monkeypatch.setenv("WRS_AGENT_TOKEN", "dynamic-node-test-session-token")
    with (tmp_path / "router.log").open("wb") as log:
        process = await asyncio.to_thread(
            subprocess.Popen,
            [str(router), "-c", str(config)], stdout=log, stderr=subprocess.STDOUT,
            creationflags=NO_WINDOW,
        )
        try:
            async with asyncio.timeout(5):
                while True:
                    assert process.poll() is None, "router exited during startup"
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", port)
                    except OSError:
                        await asyncio.sleep(0.02)
                    else:
                        writer.close()
                        await writer.wait_closed()
                        break
            yield dict(endpoint=endpoint, site="local", env_id="dynamic-" + new_id()[:12])
        finally:
            process.terminate()
            await asyncio.to_thread(process.wait, timeout=5)


@pytest.fixture
async def start_node(network, tmp_path):
    workers = []

    def start(cls, name, **overrides):
        options = dict(network, node_id=name, journal=tmp_path / f"{name}-{len(workers)}.db")
        options.update(overrides)
        node = cls(**options)
        task = asyncio.create_task(node._serve())
        workers.append(task)
        return node, task

    yield start
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


async def stop_node(system, name, worker):
    await system.registry.transport(name).request(
        f"request/node/{name}/shutdown", {}, control=True,
    )
    await asyncio.wait_for(worker, 3)
    await eventually(system.nodes, lambda rows: rows[name]["health"] == "offline")


@pytest.mark.parametrize("missed_notifications", [False, True])
async def test_existing_agent_and_client_accept_an_unlisted_late_node(
    network, start_node, monkeypatch, missed_notifications,
):
    original_change = NodeRegistry._change
    blocked = set()

    def change(view, sample, *, historical=False, query=None):
        if view in blocked and not historical:
            return  # Simulate subscriber loss; liveliness queries still use the real router.
        return original_change(view, sample, historical=historical, query=query)

    monkeypatch.setattr(NodeRegistry, "_change", change)
    agent, _ = start_node(AgentNode, "agent")
    async with System.connect(**network) as system:
        await system.registry.wait_for("agent")
        assert set(await system.nodes()) == {"agent"}
        assert await system.skills() == []
        # Initialize the Agent's own registry before the worker joins. The client's
        # directory being current says nothing about what the Planner can discover.
        assert set(await system.agent.request("request/nodes", {})) == {"agent"}
        if missed_notifications:
            blocked.add(agent.registry)
        worker, _ = start_node(CounterNode, "late-worker")
        await system.registry.wait_for("late-worker")
        assert {item.name for item in await system.skills()} == {"count"}
        await eventually(
            lambda: system.agent.request("request/nodes", {}),
            lambda rows: rows.get("late-worker", {}).get("ready"),
        )
        direct = await system.action("count")
        assert (await direct.wait()).state == "SUCCEEDED"
        task = await system.start(step("count"))
        assert (await task.wait()).state == "SUCCEEDED"
        assert worker.executor.executions == 2


async def test_restart_rejects_old_authority_and_does_not_resume_a_waiting_step(
    network, start_node,
):
    start_node(AgentNode, "agent")
    gate, _ = start_node(GateNode, "gate")
    _, original_worker = start_node(CounterNode, "counter")
    async with System.connect(**network) as system:
        for name in ("agent", "gate", "counter"):
            await system.registry.wait_for(name)
        old = await system.clients["counter"].context()
        stale = ActionRequest(
            action_id=new_id(), task_id=new_id(), task_revision=0,
            boot_id=old.boot_id, control_epoch=old.control_epoch,
            lease_id=old.lease_id, state_version=old.state_version, skill="count", args={},
        )
        first = step("gate")
        pending = await system.start(first, step("count", after=first))
        await asyncio.wait_for(gate.entered.wait(), 3)
        await stop_node(system, "counter", original_worker)
        replacement, _ = start_node(CounterNode, "counter")
        await system.registry.wait_for("counter")
        fresh = await system.clients["counter"].context()
        assert fresh.boot_id != old.boot_id
        gate.release.set()
        result = await pending.wait()
        assert result.state in {"FAILED", "CANCELLED", "UNKNOWN"}
        assert replacement.executor.executions == 0
        receipt = await submit_request(system.clients["counter"], stale)
        assert not receipt.accepted and receipt.reason == "stale_boot"
        task = await system.start(step("count"))
        assert (await task.wait()).state == "SUCCEEDED"
        assert replacement.executor.executions == 1


async def test_offline_owner_does_not_silently_select_a_new_provider(network, start_node):
    _, original = start_node(CounterNode, "first")
    async with System.connect(**network) as system:
        await system.registry.wait_for("first")
        direct = await system.action("count")
        assert (await direct.wait()).state == "SUCCEEDED"
        await stop_node(system, "first", original)
        backup, _ = start_node(CounterNode, "backup")
        await system.registry.wait_for("backup")
        with pytest.raises(AgentError, match="skill_provider_ambiguous|node_unavailable"):
            await system.action("count")
        assert backup.executor.executions == 0
        # Explicit selection is a separate user decision, not an availability fallback.
        async with System.connect(**network, skill_bindings={"count": "backup"}) as selected:
            await selected.registry.wait_for("backup")
            action = await selected.action("count")
            assert (await action.wait()).state == "SUCCEEDED"
        assert backup.executor.executions == 1


async def test_discovery_is_scoped_to_the_environment(network, start_node):
    start_node(CounterNode, "first")
    isolated = dict(network, env_id=network["env_id"] + "-other")
    start_node(CounterNode, "second", env_id=isolated["env_id"])
    async with System.connect(**network) as first, System.connect(**isolated) as second:
        await first.registry.wait_for("first")
        await second.registry.wait_for("second")
        assert set(await first.nodes()) == {"first"}
        assert set(await second.nodes()) == {"second"}
        assert (await (await first.action("count")).wait()).state == "SUCCEEDED"
        assert (await (await second.action("count")).wait()).state == "SUCCEEDED"


@pytest.mark.parametrize("second_name", ["owner", "another"])
async def test_local_duplicate_identity_or_action_address_cannot_start(
    network, start_node, tmp_path, second_name,
):
    owner, _ = start_node(CounterNode, "owner", suffix="-shared")
    async with System.connect(**network) as system:
        await system.registry.wait_for("owner")
        conflicting = CounterNode(
            **network, node_id=second_name, suffix="-shared",
            journal=Path(tmp_path) / "conflict.db",
        )
        with pytest.raises(RuntimeError, match="node_already_running"):
            await conflicting._serve()
        assert owner.executor.executions == 0
        assert set(await system.nodes()) == {"owner"}


@pytest.mark.parametrize(("remote_name", "error"), [
    ("owner", "node_ambiguous"),
    ("another", "node_endpoint_ambiguous"),
])
async def test_registry_rejects_remote_identity_and_action_address_conflicts(
    network, start_node, tmp_path, remote_name, error,
):
    owner, _ = start_node(CounterNode, "owner")
    async with System.connect(**network, skill_bindings={"count": "owner"}) as system:
        await system.registry.wait_for("owner")
        # A remote machine would not share the local instance lock. Publish a real
        # descriptor and presence directly to exercise the receiving registry.
        remote = Transport(
            network["endpoint"], network["site"], owner.target,
            "dynamic-node-test-session-token", "remote-owner",
        )
        executor = ActionExecutor(
            tmp_path / "remote.db", state=SpeechState(), backend="test",
            skills=[skill("count", count)],
            features_extra={"robot_controls": False, "controller_flush": False},
        )
        try:
            register_node(remote, remote_name, "custom", executor, env_id=network["env_id"])
            await eventually(
                system.nodes,
                lambda rows: (rows.get("owner", {}).get("error") or {}).get("code") == error,
            )
            with pytest.raises(AgentError, match=error):
                await system.action("count")
            assert owner.executor.executions == executor.executions == 0
        finally:
            await remote.close()
            await executor.close()
        await system.registry.wait_for("owner")
        action = await system.action("count")
        assert (await action.wait()).state == "SUCCEEDED"
