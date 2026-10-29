"""读真机关节角，顺带算 TCP / 法兰 / 相机的世界坐标，方便示教记录。

不加 --teach 时只读，不下发运动。--teach 会松开电机让你摆位，记完再保持。

    python fafu_real/robot/read_joints.py                 # 读一次
    python fafu_real/robot/read_joints.py --watch         # 一直读，手推机械臂看数变化（Ctrl+C 退出）
    python fafu_real/robot/read_joints.py --save look     # 读一次并记成命名位形 look
    python fafu_real/robot/read_joints.py --teach         # 逐点摆臂，回车写入 kitchen_scene.py
    python fafu_real/robot/read_joints.py --step pick_noodles --point grasp
    python fafu_real/robot/read_joints.py --dry-run       # 不连硬件，读虚拟臂（DRY_RUN_START_POSE）

记录下来的命名位形可以直接用：RUN_MODE="goto_pose" + MOVE_POSE="look"，或技能
move_named_pose(pose="look")；B / C 两个名字是 place 技能的放置位。
煮面关键点用 --teach 或 --step/--point 写进 kitchen_scene.py，不用手贴。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_REAL = str(Path(__file__).resolve().parent.parent)
if _REAL not in sys.path:
    sys.path.insert(0, _REAL)

import _paths  # noqa: F401
import fafu_config as C
import handeye
import numpy as np


def _rpy_deg(rot) -> np.ndarray:
    from wrs.basis.robot_math import rotmat_to_euler

    return np.degrees(rotmat_to_euler(np.asarray(rot, dtype=float)))


def _pose_line(name, T) -> str:
    p, rpy = T[:3, 3], _rpy_deg(T[:3, :3])
    return (
        f"{name} xyz [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m   "
        f"rpy [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}]°"
    )


def describe(q, kin, limits, camera_mat=None) -> str:
    deg = [math.degrees(v) for v in q]
    lines = [
        "关节(°)   " + "  ".join(f"J{i + 1} {v:+8.2f}" for i, v in enumerate(deg)),
        "关节(rad) " + "  ".join(f"{v:+.4f}" for v in q),
    ]
    outside = [
        f"J{i + 1}"
        for i, (v, (lo, hi)) in enumerate(zip(q, limits, strict=True))
        if not lo <= v <= hi
    ]
    if outside:
        lines.append(f"[warn] 超出 robot.cfg 软限位：{', '.join(outside)}")
    tcp, flange = kin.tcp(q), kin.flange(q)
    lines.append(_pose_line("TCP ", tcp))
    lines.append(
        f"     夹爪朝向 z=[{tcp[0, 2]:+.3f}, {tcp[1, 2]:+.3f}, {tcp[2, 2]:+.3f}]"
        f"（竖直朝下时约为 [0, 0, -1]）"
    )
    lines.append(_pose_line("法兰", flange))
    if camera_mat is not None:
        cam = handeye.camera_pose(flange, camera_mat)
        lines.append(_pose_line("相机", cam))
    pasted = "(" + ", ".join(f"{v:.2f}" for v in deg) + ")"
    lines.append("命名位形可粘贴到 fafu_config.NAMED_POSES_DEG：" + pasted)
    lines.append("煮面关键点用 --teach 或 --step 步骤 --point 点名 写入 kitchen_scene.py：" + pasted)
    return "\n".join(lines)


def save_pose(name: str, q) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name):
        raise SystemExit(f"位形名只能用字母数字和 _ . -：{name!r}")
    poses = {}
    if os.path.isfile(C.TAUGHT_POSES_JSON):
        with open(C.TAUGHT_POSES_JSON, encoding="utf-8") as f:
            poses = json.load(f)
    old = poses.get(name)
    poses[name] = [round(math.degrees(v), 3) for v in q]
    with open(C.TAUGHT_POSES_JSON, "w", encoding="utf-8") as f:
        json.dump(poses, f, ensure_ascii=False, indent=2)
    note = f"（覆盖了旧值 {old}）" if old is not None else ""
    return f"已记录 {name} = {poses[name]} → {C.TAUGHT_POSES_JSON}{note}"


_SCENE = Path(__file__).resolve().parent / "kitchen_scene.py"
_POINT_HINT = {
    "grasp": "到达后会按这个物体合爪",
    "release": "到达后会张开，物体留下",
    "tilt": "到达后停一会儿，漏勺里的东西记到目标上",
    "upright": "这一下把漏勺口带回朝上",
}


def format_deg(q) -> str:
    return "(" + ", ".join(f"{math.degrees(v):.2f}" for v in q) + ")"


def save_scene_point(step_id: str, point: str, q, path: Path = _SCENE) -> str:
    """把当前关节角写进 kitchen_scene.py 里这一步的这个点，覆盖旧值。"""
    import kitchen_scene

    if step_id not in kitchen_scene.STEPS:
        known = "、".join(kitchen_scene.STEPS)
        raise SystemExit(f"没有步骤 {step_id}。可选：{known}")
    names = [name for name, _deg in kitchen_scene.STEPS[step_id]["points"]]
    if point not in names:
        raise SystemExit(f"{step_id} 没有点 {point}。可选：{'、'.join(names)}")
    text = path.read_text(encoding="utf-8")
    block = re.compile(
        rf'("{re.escape(step_id)}"\s*:\s*\{{.*?"points"\s*:\s*\[)(.*?)(\n[ \t]*\],)',
        re.S,
    )
    found = block.search(text)
    if found is None:
        raise SystemExit(f"{path.name} 里找不到 {step_id} 的 points")
    point_re = re.compile(
        rf'(\("{re.escape(point)}"\s*,\s*)(?:None|\([^)\n]*\))(\s*\),)'
    )
    rendered = format_deg(q)
    body, count = point_re.subn(rf"\g<1>{rendered}\2", found.group(2), count=1)
    if count != 1:
        raise SystemExit(f"{step_id} 里找不到可替换的点 {point}")
    path.write_text(text[: found.start(2)] + body + text[found.end(2) :], encoding="utf-8")
    label = kitchen_scene.STEPS[step_id]["label"]
    return f"已写入 {label} · {point} = {rendered} → {path.name}"


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        print()
        return "q"


def teach(arm) -> None:
    """逐个关键点：松开电机，人把臂摆到位，回车后把关节角写进 kitchen_scene.py。"""
    import kitchen_scene

    todo = [
        (step_id, step["label"], name, deg)
        for step_id, step in kitchen_scene.STEPS.items()
        for name, deg in step["points"]
    ]
    print(f"共 {len(todo)} 个关键点，按煮面顺序来。")
    print("每个点：先把手放在臂上，回车后电机松开（会下坠），摆到位再回车写入。")
    print("s 回车跳过，q 回车结束。")
    for index, (step_id, label, name, old) in enumerate(todo, start=1):
        hint = _POINT_HINT.get(name, "只移动，夹爪不动")
        print(f"\n[{index}/{len(todo)}] {label} · {name}（{step_id}） {hint}")
        if old is not None:
            print(f"  文件里已有 {old}，写入会覆盖")
        answer = _ask("  回车松开并摆这个点，s 跳过，q 结束：")
        if answer == "q":
            print("停止示教")
            return
        if answer == "s":
            continue
        if arm.execute:
            arm.release_to_move()
            answer = _ask("  摆好后回车写入，s 放弃这个点，q 结束：")
            q = arm.capture_and_hold()
        else:
            answer = _ask("  dry-run：回车把当前虚拟关节角写入，s 跳过，q 结束：")
            q = arm.joint_values()
        if answer == "q":
            print("停止示教")
            return
        if answer == "s":
            continue
        print(save_scene_point(step_id, name, q))
    print("\n关键点问完了。")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="只读真机关节角 + TCP/法兰/相机坐标")
    p.add_argument("--watch", action="store_true", help="持续读取")
    p.add_argument("--hz", type=float, default=2.0, help="--watch 的刷新频率")
    p.add_argument("--save", metavar="NAME", help="把当前关节角记成命名位形")
    p.add_argument("--teach", action="store_true", help="逐点摆臂，回车写入 kitchen_scene.py")
    p.add_argument("--step", help="写入 kitchen_scene.py 的步骤名，例如 pick_noodles")
    p.add_argument("--point", help="这一步里的点名，例如 grasp")
    p.add_argument("--dry-run", action="store_true", help="不连硬件，读虚拟臂")
    args = p.parse_args(argv)
    if args.teach and (args.watch or args.save or args.step or args.point):
        raise SystemExit("--teach 单独用，不要和 --watch / --save / --step 一起")
    if (args.step is None) != (args.point is None):
        raise SystemExit("--step 和 --point 要一起给，例如 --step pick_noodles --point grasp")

    from arm import Arm
    from kinematics import Kinematics

    arm = Arm(execute=not args.dry_run)
    kin = Kinematics(arm.joint_limits)
    camera_mat = handeye.try_load()
    if camera_mat is None and not args.teach:
        print(f"[info] 没有有效手眼标定（{C.HANDEYE_JSON}），不打印相机位姿")
    arm.connect()
    try:
        if args.teach:
            teach(arm)
            return 0
        if args.watch:
            period = 1.0 / max(0.2, args.hz)
            while True:
                print("\n" + time.strftime("%H:%M:%S"))
                print(describe(arm.joint_values(), kin, arm.joint_limits, camera_mat))
                time.sleep(period)
        q = arm.joint_values()
        print(describe(q, kin, arm.joint_limits, camera_mat))
        if args.save:
            print(save_pose(args.save, q))
        if args.step:
            print(save_scene_point(args.step, args.point, q))
    except KeyboardInterrupt:
        print("\n停止读取")
    finally:
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
