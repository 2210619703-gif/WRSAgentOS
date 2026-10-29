"""煮面每一步的关键关节角（度，J1~J6）。

摆好机械臂后让程序自己写入，不用手贴：

    python fafu_real/robot/read_joints.py --teach
    python fafu_real/robot/read_joints.py --step pick_noodles --point grasp

``--teach`` 会按下面的点一个一个问。回车后电机松开，扶着臂摆到位，再回车就写进这个文件。
``None`` 表示还没记。

留下的点都要填。不用的点整行删掉，程序按列表顺序做关节插值。
点的名字决定夹爪，这些名字不要改：

- ``grasp``：到达后按这个物体合爪
- ``release``：到达后张开，物体留下
- ``tilt``：到达后停一会儿，把漏勺里的东西记到目标上
- ``upright``：这一下把漏勺口带回朝上
- 其他名字：只移动

同一技能出现两次（桌上抓漏勺、从锅里捞出）各记各的。
关节角还全空时，这一步仍按物体位置规划；填了一个就必须填完。

下面的 ``OBJECTS`` 只留物体名字给世界状态用（手里拿着谁、放进哪个容器），不再记位置。
"""

from __future__ import annotations

STEPS = {
    "pick_noodles": {
        "label": "抓面条",
        "skill": "pick",
        "args": {"object": "noodles"},
        "points": [
            ("approach", None),
            ("grasp", None),
            ("lift", None),
        ],
    },
    "noodles_into_strainer": {
        "label": "面条放进漏勺",
        "skill": "seat_in",
        "args": {"object": "noodles", "container": "strainer"},
        "points": [
            ("above", None),
            ("release", None),
            ("retreat", None),
        ],
    },
    "pick_strainer": {
        "label": "抓漏勺",
        "skill": "pick",
        "args": {"object": "strainer"},
        "when": "out_of_pot",
        "points": [
            ("approach", None),
            ("grasp", None),
            ("lift", None),
        ],
    },
    "strainer_into_pot": {
        "label": "漏勺放进煮面锅",
        "skill": "seat_in",
        "args": {"object": "strainer", "container": "pot"},
        "points": [
            ("above", None),
            ("release", None),
            ("retreat", None),
        ],
    },
    "pick_strainer_out": {
        "label": "从锅里捞出漏勺",
        "skill": "pick",
        "args": {"object": "strainer"},
        "when": "in_pot",
        "points": [
            ("approach", None),
            ("grasp", None),
            ("lift", None),
        ],
    },
    "shake_strainer": {
        "label": "沥水",
        "skill": "shake_off",
        "args": {"object": "strainer"},
        "points": [
            ("up", None),
            ("left", None),
            ("right", None),
            ("center", None),
        ],
    },
    "pour_to_plate": {
        "label": "倒到盘子",
        "skill": "pour",
        "args": {"source": "strainer", "target": "plate"},
        "points": [
            ("above", None),
            ("tilt", None),
            ("upright", None),
        ],
    },
    "strainer_onto_table": {
        "label": "漏勺放回桌上",
        "skill": "place_on",
        "args": {"object": "strainer", "surface": "table"},
        "points": [
            ("above", None),
            ("release", None),
            ("retreat", None),
        ],
    },
}

# 例：
#     ("grasp", (9.33, 71.74, 78.36, -85.76, 0.41, 13.11)),

OBJECTS = {
    "table": {"label": "桌面", "kind": "table"},
    "cooktop": {"label": "电磁炉", "kind": "appliance"},
    "pot": {"label": "煮面锅", "kind": "container"},
    "strainer": {"label": "漏勺", "kind": "container"},
    "noodles": {"label": "面条", "kind": "food"},
    # 权重只有一个 bowl 类。对得上 bowl_support.stl 的是生食碗，另一只是盛出碗。
    # 都对不上时，Y 较大的当作生食碗，Y 较小的当作盛出碗。
    "bowl_raw": {"label": "生食碗", "kind": "container"},
    "bowl_serve": {"label": "盛出碗", "kind": "container"},
    "plate": {"label": "盘子", "kind": "surface"},
    "spoon": {"label": "汤勺", "kind": "tool"},
}


def _when_ok(code: str, world) -> bool:
    container = world.items["strainer"].container
    if code == "out_of_pot":
        return container != "pot"
    if code == "in_pot":
        return container == "pot"
    raise ValueError(f"不认识的步骤条件 {code!r}")


def match_steps(skill: str, args, world) -> list[dict]:
    """技能名和参数对得上、并且符合当前世界状态的示教步骤。"""
    got = args.model_dump() if hasattr(args, "model_dump") else dict(args)
    hits = []
    for step_id, step in STEPS.items():
        if step["skill"] != skill:
            continue
        if any(got.get(key) != value for key, value in step["args"].items()):
            continue
        code = step.get("when")
        if code and not _when_ok(code, world):
            continue
        hits.append({**step, "id": step_id})
    return hits


def point_gap(step: dict) -> list[str]:
    """还没填的点。空列表表示可以按示教走；全部都空则调用方继续用逆解。"""
    return [name for name, deg in step["points"] if deg is None]
