"""厨房技能的轨迹拖影（ghost trajectory / pose sweep），在 Panda3D 窗口里看。

沿一条已经规划成功的关节轨迹采样少量构型，把机械臂网格半透明叠在一起。
颜色用各连杆自己的本色，从浅收到深、透明度也跟着加深。这段里被抓住并
移动的物体用同样的采样叠出拖影，颜色用物体自己的本色。抓取 pickle 见
write_object_grasps.py。

    python kitchen_paths/render_skill_sweeps.py --show pick_bowl_support_2

左键旋转，滚轮缩放，Esc 关闭。不打开浏览器。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kitchen_paths.panda_view import capture, show
from wrs_agent.env.kitchen_motion import KitchenMotion, jaw_mesh
from wrs_agent.env.wrs import ROOT, VirtualModel
from wrs_agent.scene import load_scene

SAMPLES = 8

# 已经能规划成功、并且法兰会走一段路的技能。observe 只刷新状态，
# set_gripper 只开合夹爪，都没有关节轨迹，不做成拖影。
SEQUENCE = (
    ("pick", {"object": "bowl_support_2"}, "pick_bowl_support_2", "pick 第二套碗"),
    ("pour", {"source": "bowl_support_2", "target": "skimmer"}, "pour_bowl_to_skimmer", "pour 碗倒进漏勺"),
    ("place_on", {"object": "bowl_support_2", "surface": "bowl_rack_2"}, "place_on_bowl_rack_2", "place_on 碗放回碗架"),
    ("pick", {"object": "skimmer"}, "pick_skimmer", "pick 漏勺"),
    ("seat_in", {"object": "skimmer", "container": "pot"}, "seat_in_pot", "seat_in 漏勺放进锅"),
    ("pick", {"object": "skimmer"}, "pick_skimmer_from_pot", "pick 从锅里再抓漏勺"),
    ("shake_off", {"object": "skimmer"}, "shake_off", "shake_off 抖漏勺"),
    ("pour", {"source": "skimmer", "target": "bowl_support"}, "pour_skimmer_to_bowl", "pour 漏勺倒进碗"),
    ("place_on", {"object": "skimmer", "surface": "skimmer_holder"}, "place_on_skimmer_holder", "place_on 漏勺放回架子"),
)


def subsample(frames, count=SAMPLES):
    if len(frames) <= count:
        return list(frames)
    index = np.unique(np.rint(np.linspace(0, len(frames) - 1, count)).astype(int))
    return [frames[int(i)] for i in index]


def snapshot_frame(frame):
    return {
        "qs": np.asarray(frame["qs"], dtype=float).copy(),
        "jaw": float(frame.get("jaw", 0.0)),
        "jaw_gap": float(frame.get("jaw_gap", 0.016)),
        "poses": {
            name: (np.asarray(pos, dtype=float).copy(), np.asarray(rot, dtype=float).copy())
            for name, (pos, rot) in frame["poses"].items()
        },
        "held": frame.get("held"),
    }


def scene_poses(motion):
    return {name: motion._pose(name) for name in motion.model.objects}


def play(motion, name, args, stem, title):
    """Plan, keep the sampled frames, then actually step so the next skill starts there."""
    poses = scene_poses(motion)
    count = motion.plan_skill(name, args)
    frames = subsample([snapshot_frame(frame) for frame in motion.path])
    for index in range(count):
        motion.step(index)
    if not motion.verify():
        raise RuntimeError(f"{stem} 规划后校验失败")
    return {"stem": stem, "title": title, "frames": frames, "poses": poses}


def kitchen():
    model = VirtualModel()
    model.load_scene(load_scene(ROOT / "wrs_agent/assets/kitchen/scene.toml"))
    motion = KitchenMotion(model)
    motion.model.apply([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return motion


def matrix_of(pos, rot):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def deepen(rgb, blend):
    """Keep the arm's own hue; early samples lighten toward white, later ones darken."""
    base = np.asarray(rgb, dtype=float)
    light = np.clip(base * 1.2 + 0.22, 0.0, 1.0)
    dark = np.clip(base * 0.42, 0.0, 1.0)
    return light + (dark - light) * blend


def pose_moved(start, frames, name):
    origin = start.get(name)
    if origin is None:
        return False
    for frame in frames:
        pose = frame["poses"].get(name)
        if pose is None:
            continue
        if float(np.linalg.norm(pose[0] - origin[0])) > 1e-4:
            return True
        if not np.allclose(pose[1], origin[1], atol=1e-3):
            return True
    return False


def stamp_batches(motion, segment):
    """Still objects stay put. A grasped object that moves is ghosted with the arm."""
    batches = []
    marks = []
    frames = segment["frames"]
    moving = {
        name for name in motion.model.objects
        if pose_moved(segment["poses"], frames, name)
    }
    for name, (data, obj) in motion.model.objects.items():
        if name in moving:
            continue
        pos, rot = segment["poses"].get(name, motion._pose(name))
        obj.set_pos_rotmat(pos=pos, rotmat=rot)
        obj.rgb = data.rgb
        batches.extend(capture(obj))
    for index, frame in enumerate(frames):
        blend = 0.0 if len(frames) <= 1 else index / (len(frames) - 1)
        alpha = 0.16 + 0.72 * blend
        motion._jaw_visual.geom = jaw_mesh(
            motion._jaw_open, motion._jaw_mid, motion._jaw_closed,
            frame["jaw"], frame["jaw_gap"],
        )
        motion.model.apply(frame["qs"])
        tip = next(link for link in motion.model.robot.runtime_lnks if link.name == "link6")
        marks.append(tip.pos.copy())
        for link in motion.model.robot.runtime_lnks:
            if not link.visuals:
                continue
            rgb = deepen(link.visuals[0].rgb, blend)
            batches.extend(capture(link, rgb, alpha))
        for name in moving:
            data, obj = motion.model.objects[name]
            pos, rot = frame["poses"].get(name, segment["poses"][name])
            obj.set_pos_rotmat(pos=pos, rotmat=rot)
            obj.rgb = data.rgb
            batches.extend(capture(obj, deepen(data.rgb, blend), alpha))
    return batches, marks


def prepare():
    motion = kitchen()
    # 这两段只取样、不执行，做饭序列仍从关节零位开始。
    segments = []
    for name, args, stem, title in (
        ("move_named_pose", {"pose": "home"}, "move_named_pose", "move_named_pose 零位到 home"),
        ("move_relative", {"dx": 0.02, "dy": 0.0, "dz": 0.0}, "move_relative", "move_relative 世界 +X 2 cm"),
    ):
        poses = scene_poses(motion)
        motion.plan_skill(name, args)
        segments.append({
            "stem": stem,
            "title": title,
            "frames": subsample([snapshot_frame(frame) for frame in motion.path]),
            "poses": poses,
        })
        print("planned", stem, "frames", len(segments[-1]["frames"]))
    for name, args, stem, title in SEQUENCE:
        segments.append(play(motion, name, args, stem, title))
        print("planned", stem, "frames", len(segments[-1]["frames"]))
    return motion, {item["stem"]: item for item in segments}


def main():
    parser = argparse.ArgumentParser(description="在 Panda3D 里看厨房技能的轨迹拖影")
    parser.add_argument("--show", metavar="STEM", help="打开这一段的 Panda3D 窗口")
    parser.add_argument("--list", action="store_true", help="只列出可查看的段名")
    parser.add_argument("--shot", help=argparse.SUPPRESS)
    args = parser.parse_args()
    motion, segments = prepare()
    if args.list:
        for stem, item in segments.items():
            print(f"{stem}\t{item['title']}\t{len(item['frames'])} 帧")
        return
    if args.show is None:
        for stem, item in segments.items():
            _batches, marks = stamp_batches(motion, item)
            shift = float(np.linalg.norm(marks[-1] - marks[0]))
            print(f"sweep {stem}: {len(item['frames'])} poses, travel {shift:.3f} m")
        print("查看某一段: python kitchen_paths/render_skill_sweeps.py --show <名字>")
        return
    segment = segments.get(args.show)
    if segment is None:
        names = ", ".join(segments)
        raise SystemExit(f"没有这段: {args.show}。可选: {names}")
    batches, _marks = stamp_batches(motion, segment)
    show(
        batches,
        f"轨迹拖影 · {segment['title']}",
        look=(0.3, 0.0, 0.35),
        eye=(1.65, -1.85, 1.55),
        shot=args.shot,
    )


if __name__ == "__main__":
    main()
