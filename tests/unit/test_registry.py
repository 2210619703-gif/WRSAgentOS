import asyncio
from types import SimpleNamespace

import pytest
import zenoh
from skill_fixtures import features as skill_features

from wrs_agent import step
from wrs_agent.bindings import DEFAULT, load_bindings
from wrs_agent.registry import NodeRegistry
from wrs_agent.schemas import NodeInfo
from wrs_agent.transport import RemoteError


class QueryFixture:
    """Unit-only Zenoh boundary; cross-process discovery is tested separately."""

    def __init__(self, info=None):
        self.info = info
        self.infos = {info.node_id: info} if info else {}
        self.endpoint, self.site, self.env_id = "unit", "local", "room"
        self.token, self.source = "unit-session-token", "reader"
        self.session = SimpleNamespace(liveliness=lambda: self)
        self.handles = []
        self.live = {(info.node_id, info.target, info.boot_id)} if info else set()
        self.callback = None
        self.before_reply = self.before_initial_reply = None
        self.queries = 0
        self.presence_queries = 0
        self.closed, self.unsubscribed = [], False

    def key(self, suffix):
        return f"wrs/v4/local/{self.env_id}/{suffix}"

    def declare_subscriber(self, key, handler, *, history):
        assert not history
        assert key == "wrs/v4/local/room/discovery/*/*/*"
        self.callback = handler.callback

        def undeclare():
            self.unsubscribed = True

        return SimpleNamespace(undeclare=undeclare)

    @staticmethod
    def sample(identity, present=True):
        return SimpleNamespace(
            key_expr="wrs/v4/local/room/discovery/" + "/".join(identity),
            kind=zenoh.SampleKind.PUT if present else zenoh.SampleKind.DELETE,
        )

    def get(self, key, handler, *, timeout, cancellation_token):
        self.presence_queries += 1
        history = self.live.copy()
        if self.before_initial_reply:
            self.before_initial_reply()
        for identity in history:
            handler.callback(SimpleNamespace(ok=self.sample(identity)))
        handler.drop()

    def emit(self, boot, present, *, node_id="speaker", target="room"):
        identity = (node_id, target, boot)
        self.live.add(identity) if present else self.live.discard(identity)
        self.callback(self.sample(identity, present))

    def remote(self, endpoint, site, target, token, source):
        assert (endpoint, site, token, source) == (
            self.endpoint, self.site, self.token, self.source,
        )

        async def close():
            self.closed.append(target)

        return SimpleNamespace(env_id=target, request=self.request, close=close)

    async def request(self, key, payload, *, timeout):  # noqa: ASYNC109 - query fixture
        self.queries += 1
        name = key.removeprefix("request/node/")
        info = self.info if name == "speaker" else self.infos.get(name)
        if info is None:
            raise TimeoutError
        result = info.model_dump()
        if self.before_reply:
            self.before_reply()
        return result


@pytest.fixture
def network(monkeypatch):
    def create(info=None):
        bus = QueryFixture(info)
        monkeypatch.setattr("wrs_agent.registry.Transport", bus.remote)
        return bus

    return create


def speaker(boot="first", **changes):
    return NodeInfo(
        node_id="speaker", node_type="tts", target="room", actions=True,
        boot_id=boot, ready=True, health="ready", skills={"speak": 1}, features=["speak"],
    ).model_copy(update=changes)


def registry(bus):
    return NodeRegistry(bus, bindings={"speak": "speaker"})


async def test_registry_presence_offline_restart_and_explicit_binding(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    assert view.node_for("speak") == "speaker"
    assert not hasattr(view, "checked")
    original_client = view.clients["speaker"]
    bus.emit("first", False)
    assert view.snapshot()["speaker"]["health"] == "offline"
    assert view.clients["speaker"] is original_client
    with pytest.raises(ValueError, match="node_unavailable"):
        view.node_for("speak")
    bus.info = None
    before = bus.queries
    assert (await view.refresh())["speaker"]["health"] == "offline"
    assert bus.queries == before
    bus.info = speaker("second", skills={}, features=[])
    bus.emit("second", True)
    await view.refresh()
    with pytest.raises(ValueError, match="skill_not_on_node"):
        view.node_for("speak")
    assert view.entries["speaker"].boot_id == "second"
    bus.info = bus.info.model_copy(update={"node_id": "impostor"})
    assert not (await view.refresh())["speaker"]["ready"]


async def test_presence_is_not_readiness_and_feature_cache_tracks_instance(network):
    bus = network(speaker())
    view = registry(bus)
    calls = 0

    async def features():
        nonlocal calls
        calls += 1
        return skill_features(
            ["speak"], robot_controls=False, boot_id=bus.info.boot_id,
            skill_revision=bus.info.skill_revision,
        )

    client = SimpleNamespace(features=features)
    await view.refresh()
    await view.features("speaker", client)
    await view.refresh()
    await view.features("speaker", client)
    assert calls == 1
    bus.info = speaker(ready=False, health="held")
    assert (await view.refresh())["speaker"]["health"] == "held"
    with pytest.raises(ValueError, match="not_ready"):
        view.node_for("speak")
    assert await view.wait_for(role="tts") == "speaker"
    bus.emit("first", False)
    bus.emit("second", True)
    bus.info = speaker("second")
    await view.refresh()
    await view.features("speaker", client)
    assert calls == 2
    with pytest.raises(ValueError, match="instance_changed"):
        view.check_instance("speaker", "first")
    bus.info = None
    offline_app = (await view.refresh())["speaker"]
    assert offline_app["boot_id"] == "second" and offline_app["health"] == "unknown"
    assert not offline_app["ready"]


async def test_duplicate_instances_and_descriptor_leave_race_fail_closed(network):
    bus = network(speaker())
    view = registry(bus)
    bus.before_reply = lambda: bus.emit("first", False)
    assert (await view.refresh())["speaker"]["health"] == "offline"
    bus.before_reply = None
    bus.emit("first", True)
    bus.emit("second", True)
    assert (await view.refresh())["speaker"]["health"] == "unknown"
    with pytest.raises(ValueError, match="node_ambiguous"):
        view.node_for("speak")
    bus.emit("second", False)
    assert (await view.refresh())["speaker"]["ready"]
    for n in range(17):
        bus.emit(f"extra-{n}", True)
    assert view.snapshot()["speaker"]["health"] == "unknown"


async def test_initial_query_cannot_revive_a_concurrent_departure(network):
    bus = network(speaker())
    view = registry(bus)
    bus.before_initial_reply = lambda: bus.emit("first", False)
    assert await view.refresh() == {}
    assert bus.queries == 0


async def test_refresh_reconciles_missed_presence_events_with_a_bounded_frequency(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    assert bus.presence_queries == 1
    # The router has the token, but this long-lived subscriber missed its notification.
    bus.infos["late"] = speaker(node_id="late", target="room-late", skills={})
    bus.live.add(("late", "room-late", "first"))
    assert "late" not in await view.refresh()
    assert bus.presence_queries == 1
    view._last_presence = float("-inf")
    assert (await view.refresh())["late"]["ready"]
    assert bus.presence_queries == 2

    # Missed departures and replacements must also be reconciled, not just arrivals.
    bus.live.remove(("speaker", "room", "first"))
    bus.live.add(("speaker", "room", "next"))
    bus.info = speaker("next")
    view._last_presence = float("-inf")
    assert (await view.refresh())["speaker"]["boot_id"] == "next"
    with pytest.raises(ValueError, match="node_instance_changed"):
        view.check_instance("speaker", "first")
    assert bus.presence_queries == 3


async def test_presence_resync_keeps_live_view_and_does_not_revive_a_departure(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()

    def concurrent_change():
        view.check_instance("speaker", "first")  # Not temporarily offline during get().
        bus.emit("first", False)
        with pytest.raises(ValueError, match="node_unavailable"):
            view.check_instance("speaker", "first")

    bus.before_initial_reply = concurrent_change
    view._last_presence = float("-inf")
    assert (await view.refresh())["speaker"]["health"] == "offline"


async def test_cancelled_presence_resync_preserves_view_and_can_retry(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    original_get = bus.get
    entered = asyncio.Event()
    callbacks = []

    def pending_get(key, handler, **kwargs):
        callbacks.append(handler)
        entered.set()

    bus.get = pending_get
    view._last_presence = float("-inf")
    pending = asyncio.create_task(view.refresh())
    await entered.wait()
    view.check_instance("speaker", "first")
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert view._queried is None
    view.check_instance("speaker", "first")
    bus.get = original_get
    bus.live.clear()
    # An old, cancelled query's late reply must not contaminate the new snapshot.
    bus.before_initial_reply = lambda: callbacks[0].callback(
        SimpleNamespace(ok=bus.sample(("speaker", "room", "first"))),
    )
    assert (await view.refresh())["speaker"]["health"] == "offline"


async def test_unknown_node_arrives_and_its_address_is_queried_without_bindings(network):
    bus = network()
    view = NodeRegistry(bus)
    clients, buses = view.clients, view.buses
    assert await view.refresh() == {}
    waiter = asyncio.create_task(view.wait_for(role="tts"))
    await asyncio.sleep(0)
    bus.infos["late"] = speaker(node_id="late", target="room-late")
    bus.emit("first", True, node_id="late", target="room-late")
    assert await waiter == "late"
    assert view.node_for("speak") == "late"
    assert view.transport("late").env_id == "room-late"
    assert view.clients is clients and view.buses is buses
    assert clients["late"].transport is buses["late"]
    with pytest.raises(ValueError, match="node_role_mismatch"):
        await view.wait_for("late", role="wrs")
    await view.close()
    await view.close()
    assert bus.closed == ["room-late"] and bus.unsubscribed


async def test_duplicate_action_addresses_are_rejected_but_shared_nonaction_is_allowed(network):
    bus = network(speaker())
    view = registry(bus)
    bus.infos["agent"] = NodeInfo(
        node_id="agent", node_type="agent", target="room", boot_id="a", ready=True, health="ready",
    )
    await view.refresh()
    bus.emit("a", True, node_id="agent")
    assert view.snapshot()["speaker"]["error"]["code"] == "node_not_ready"
    with pytest.raises(ValueError, match="node_not_ready"):
        view.transport("speaker")
    assert (await view.refresh())["speaker"]["ready"]
    bus.infos["other"] = speaker(node_id="other", boot_id="b")
    bus.emit("b", True, node_id="other")
    state = await view.refresh()
    assert state["speaker"]["error"]["code"] == "node_endpoint_ambiguous"
    assert state["other"]["error"]["code"] == "node_endpoint_ambiguous"
    with pytest.raises(ValueError, match="node_endpoint_ambiguous"):
        view.check_instance("speaker", "first")
    bus.emit("b", False, node_id="other")
    assert view.snapshot()["speaker"]["ready"]


async def test_descriptor_address_must_match_discovery_and_new_address_replaces_client(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    previous = view.clients["speaker"]
    bus.info = speaker(target="wrong")
    state = await view.refresh()
    assert state["speaker"]["error"]["code"] == "node_identity_mismatch"
    bus.emit("first", False)
    bus.info = speaker("next", target="room-next")
    bus.emit("next", True, target="room-next")
    await view.refresh()
    assert view.clients["speaker"] is not previous
    assert previous.transport is bus
    assert view.clients["speaker"].transport.env_id == "room-next"
    with pytest.raises(ValueError, match="node_instance_changed"):
        view.check_instance("speaker", "first")


async def test_wait_for_keeps_waiting_after_initial_descriptor_attempts_time_out(network):
    bus = network(speaker())
    view = registry(bus)
    original_request = bus.request
    calls = 0

    async def delayed_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise TimeoutError
        return await original_request(*args, **kwargs)

    bus.request = delayed_request
    assert await view.wait_for("speaker", timeout=1) == "speaker"
    assert calls == 3
    assert view.snapshot()["speaker"]["ready"]


async def test_wait_for_unanswered_descriptor_obeys_its_explicit_deadline(network):
    bus = network(speaker())
    view = registry(bus)
    calls = 0

    async def unavailable(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError

    bus.request = unavailable
    with pytest.raises(TimeoutError):
        await view.wait_for("speaker", timeout=0.02)
    assert calls >= 2
    assert view.entries["speaker"].error.code == "request_timeout"


async def test_discovery_preserves_unauthorized_but_presence_changes_take_precedence(network):
    bus = network(speaker())
    view = registry(bus)

    async def unauthorized(*args, **kwargs):
        raise RemoteError("unauthorized")

    bus.request = unauthorized
    assert (await view.refresh())["speaker"]["error"]["code"] == "unauthorized"
    with pytest.raises(ValueError, match="unauthorized"):
        view.transport("speaker")
    with pytest.raises(ValueError, match="unauthorized"):
        await view.wait_for("speaker")
    bus.emit("second", True)
    with pytest.raises(ValueError, match="node_ambiguous"):
        view.transport("speaker")
    bus.emit("first", False)
    with pytest.raises(ValueError, match="node_unavailable"):
        view.transport("speaker")


@pytest.mark.parametrize("target", ["room", "room-speaker"])
async def test_unverified_descriptor_retries_one_timeout_on_borrowed_or_new_connection(
    network, target,
):
    bus = network(speaker(target=target))
    view = registry(bus)
    original_request = bus.request
    attempts = []

    async def delayed_request(key, payload, *, timeout):  # noqa: ASYNC109 - query fixture
        attempts.append((key, timeout))
        # A route that needs 0.7 s cannot fit in two consecutive 0.3 s query windows.
        if timeout < .7:
            raise TimeoutError
        return await original_request(key, payload, timeout=timeout)

    bus.request = delayed_request
    assert (await view.refresh())["speaker"]["ready"]
    assert attempts == [("request/node/speaker", 0.3), ("request/node/speaker", 2.0)]

    async def unavailable(*args, **kwargs):
        attempts.append((args[0], kwargs["timeout"]))
        raise TimeoutError

    bus.request = unavailable
    if target != bus.env_id:
        view.buses["speaker"].request = unavailable
    state = await view.refresh()
    assert len(attempts) == 3  # Verified instances keep the existing single-read deadline.
    assert state["speaker"]["error"]["code"] == "request_timeout"


@pytest.mark.parametrize("change", ["timeout", "depart", "restart", "invalid"])
async def test_initial_descriptor_retry_is_bounded_and_keeps_instance_identity(network, change):
    bus = network(speaker())
    view = registry(bus)
    calls = 0

    async def request(key, payload, *, timeout):  # noqa: ASYNC109 - query fixture
        nonlocal calls
        calls += 1
        if change == "invalid":
            return bus.info.model_copy(update={"target": "impostor"}).model_dump()
        if calls == 1 and change in {"depart", "restart"}:
            bus.emit("first", False)
            if change == "restart":
                bus.emit("second", True)
        raise TimeoutError

    bus.request = request
    state = await view.refresh()
    assert not state["speaker"]["ready"]
    assert calls == (2 if change == "timeout" else 1)
    if change == "timeout":
        assert state["speaker"]["error"]["code"] == "request_timeout"
    assert "speaker" not in view.clients


async def test_close_during_initial_discovery_does_not_open_late_transports(network):
    bus = network(speaker(target="room-speaker"))
    started = asyncio.Event()
    callbacks = []

    def delayed_get(key, handler, *, timeout, cancellation_token):
        callbacks.append(handler)
        started.set()

    bus.get = delayed_get
    view = registry(bus)
    refresh = asyncio.create_task(view.refresh())
    await started.wait()
    await view.close()
    assert bus.unsubscribed
    callbacks[0].callback(SimpleNamespace(ok=bus.sample(next(iter(bus.live)))))
    callbacks[0].drop()
    with pytest.raises(RuntimeError, match="registry_closed"):
        await refresh
    assert view.clients == {} and view.buses == {}
    assert len(view._transports) == 1  # The borrowed source transport is still owned by Node.


async def test_close_waits_for_active_queries_and_blocks_queued_refresh(network):
    bus = network(speaker(target="room-speaker"))
    started, release = asyncio.Event(), asyncio.Event()
    original_request = bus.request

    async def held_request(*args, **kwargs):
        started.set()
        await release.wait()
        return await original_request(*args, **kwargs)

    bus.request = held_request
    view = registry(bus)
    active = asyncio.create_task(view.refresh())
    await started.wait()
    closing = asyncio.create_task(view.close())
    await asyncio.sleep(0)
    assert not closing.done() and bus.closed == []
    bus.infos["late"] = speaker(node_id="late", target="room-late")
    bus.emit("first", True, node_id="late", target="room-late")
    queued = asyncio.create_task(view.refresh())
    await asyncio.sleep(0)
    assert not queued.done()
    release.set()
    await active
    await closing
    with pytest.raises(RuntimeError, match="registry_closed"):
        await queued
    assert bus.closed == ["room-speaker"]
    assert "late" not in view.clients and "room-late" not in view._transports
    assert bus.queries == 1


async def test_same_boot_cannot_move_its_verified_service_address(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    original = view.clients["speaker"]
    bus.emit("first", False)
    bus.emit("first", True, target="room-moved")
    bus.info = speaker(target="room-moved")
    # transport reads the cached descriptor before consuming pending presence changes.
    with pytest.raises(ValueError, match="node_instance_changed"):
        view.transport("speaker")
    state = await view.refresh()
    assert state["speaker"]["error"]["code"] == "node_instance_changed"
    assert view.clients["speaker"] is original


async def test_directory_is_bounded_and_ignores_other_environment_and_malformed_keys(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    for key in (
        "wrs/v4/local/another/discovery/intruder/room/x",
        "wrs/v4/local/room/discovery/intruder/../x",
        "wrs/v4/local/room/discovery/incomplete",
    ):
        bus.callback(SimpleNamespace(key_expr=key, kind=zenoh.SampleKind.PUT))
    assert set(view.snapshot()) == {"speaker"}
    for index in range(70):
        bus.emit("boot", True, node_id=f"node{index}", target=f"room-node{index}")
    await view.refresh()
    assert len(view.snapshot()) == 64
    view.check_instance("speaker", "first")
    assert view.snapshot()["speaker"]["ready"]
    with pytest.raises(ValueError, match="discovery_capacity_exceeded"):
        await view.wait_for("node69")


def test_bindings_allow_named_providers_but_not_duplicate_action_endpoints(tmp_path):
    text = DEFAULT.read_text(encoding="utf-8").replace("nodes.wrs", "nodes.wrs_ur7e")
    text = text.replace('= "wrs"', '= "wrs_ur7e"').replace('type = "wrs_ur7e"', 'type = "wrs"')
    path = tmp_path / "bindings.toml"
    path.write_text(text, encoding="utf-8")
    nodes, bindings = load_bindings(path)
    assert bindings == {} and nodes["wrs_ur7e"]["type"] == "wrs"
    path.write_text(text.replace('suffix = "-tts"', 'suffix = ""'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_execution_endpoint"):
        load_bindings(path)


def test_small_step_api_preserves_explicit_dependencies():
    observed = step("observe")
    picked = step("pick", object="A", after=observed)
    said = step("speak", text="hello")
    assert picked.depends_on == [observed.step_id]
    assert said.depends_on == []


async def test_departed_descriptions_are_reclaimed_without_closing_frozen_clients(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    frozen = None
    for index in range(80):
        name = f"temporary{index}"
        bus.infos[name] = speaker(
            node_id=name, target="room-temporary", skills={"temporary": 1},
        )
        bus.emit("first", True, node_id=name, target="room-temporary")
        await view.refresh([name])
        frozen = frozen or view.clients[name]
        bus.emit("first", False, node_id=name, target="room-temporary")
        view.check_instance("speaker", "first")
    assert len(view.snapshot()) == 64
    assert len(view._presence()) == 1
    assert "temporary0" not in view.entries and "temporary0" not in view.clients
    assert "room-temporary" not in bus.closed
    assert frozen.transport.env_id == "room-temporary"
    with pytest.raises(ValueError, match="node_unavailable"):
        view.check_instance("temporary0", "first")
    assert view.node_for("speak") == "speaker"
    assert "temporary" in view.routes()[1]
    await view.close()
    assert bus.closed == ["room-temporary"]


async def test_refused_node_still_blocks_a_conflicting_action_address(network):
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    for index in range(63):
        name = f"peer{index}"
        bus.infos[name] = speaker(node_id=name, target="room-peers", actions=False, skills={})
        bus.emit("first", True, node_id=name, target="room-peers")
    await view.refresh()
    bus.emit("first", True, node_id="excess", target="room")
    with pytest.raises(ValueError, match="discovery_capacity_exceeded"):
        await view.wait_for("excess")
    with pytest.raises(ValueError, match="node_not_ready"):
        view.check_instance("speaker", "first")
    bus.emit("first", False, node_id="excess", target="room")
    view.check_instance("speaker", "first")


async def test_presence_overflow_reports_capacity_and_resynchronizes_after_departure(
    network, monkeypatch,
):
    monkeypatch.setattr("wrs_agent.registry.MAX_PRESENCE", 2)
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    for name in ("second", "third"):
        bus.infos[name] = speaker(node_id=name, target=f"room-{name}", actions=False, skills={})
        bus.emit("first", True, node_id=name, target=f"room-{name}")
    with pytest.raises(ValueError, match="discovery_capacity_exceeded"):
        view.check_instance("speaker", "first")
    bus.emit("first", False, node_id="second", target="room-second")
    assert (await view.refresh())["speaker"]["ready"]
    view.check_instance("speaker", "first")
    assert "third" in view.entries and not view._overflow


async def test_reclaiming_a_description_does_not_select_a_replacement_provider(network):
    bus = network(speaker())
    view = NodeRegistry(bus)
    await view.refresh()
    bus.emit("first", False)
    for index in range(64):
        name = f"peer{index}"
        bus.infos[name] = speaker(node_id=name, actions=False, skills={})
        bus.emit("first", True, node_id=name)
        await view.refresh([name])
        bus.emit("first", False, node_id=name)
    assert "speaker" not in view.entries
    assert view.routes()[0]["speak"] == "speaker"
    bus.infos["replacement"] = speaker(node_id="replacement")
    bus.emit("first", True, node_id="replacement")
    await view.refresh()
    with pytest.raises(ValueError, match="skill_provider_ambiguous"):
        view.node_for("speak")


async def test_skill_owner_memory_has_a_separate_bound_and_preserves_existing_nodes(
    network, monkeypatch,
):
    monkeypatch.setattr("wrs_agent.registry.MAX_SKILL_OWNERS", 1)
    bus = network(speaker())
    view = registry(bus)
    await view.refresh()
    bus.infos["other"] = speaker(node_id="other", target="room-other", skills={"another": 1})
    bus.emit("first", True, node_id="other", target="room-other")
    rows = await view.refresh()
    assert rows["other"]["error"]["code"] == "discovery_capacity_exceeded"
    assert rows["speaker"]["ready"] and len(view._owners) == 1


async def test_new_presence_without_a_description_is_not_a_capacity_error(network):
    bus = network()
    view = registry(bus)
    await view.refresh()
    bus.infos["late"] = speaker(node_id="late", target="room-late")
    bus.emit("first", True, node_id="late", target="room-late")
    with pytest.raises(ValueError, match="node_not_ready"):
        view.transport("late")
    assert await view.wait_for("late") == "late"
