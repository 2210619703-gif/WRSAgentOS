"""按物体各写一份抓取 pickle，并在 Panda3D 里看这次抓取。

记录的是该物体自己坐标系里的法兰位姿，外加这次抓取成功时的关节角，
所以窗口直接读 pickle，不再重规划。

    python kitchen_paths/write_object_grasps.py
    python kitchen_paths/write_object_grasps.py --show bowl_support
    python kitchen_paths/write_object_grasps.py --show bowl_support_2
    python kitchen_paths/write_object_grasps.py --show skimmer
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kitchen_paths.panda_view import capture, show
from kitchen_paths.render_skill_sweeps import kitchen, matrix_of
from wrs_agent.env.kitchen_motion import GRASP_GAP, GRASPS, jaw_mesh

HERE = Path(__file__).resolve().parent
GRASP_DIR = HERE / "grasps"


def flange_in_object(motion, frame, name):
    if name in frame["poses"]:
        pos, rot = frame["poses"][name]
    else:
        pos, rot = motion.rest[name]
    motion.model.apply(frame["qs"])
    flange = np.asarray(motion.model.tcp.tf, dtype=float)
    return np.linalg.inv(matrix_of(pos, rot)) @ flange, matrix_of(pos, rot)


def write_grasp(motion, name):
    """Plan one pick and save that object's grasp. The arm is put back afterwards."""
    from kitchen_paths.render_skill_sweeps import snapshot_frame

    motion.plan_skill("pick", {"object": name})
    contact = None
    closed = None
    for frame in motion.path:
        shot = snapshot_frame(frame)
        if shot["held"] == name and closed is None:
            closed = shot
        elif shot["held"] != name:
            contact = shot
    pose_frame = contact if contact is not None and contact["jaw"] >= 0.5 else closed
    if pose_frame is None:
        raise RuntimeError(f"{name} 没有抓取构型")
    pre_frame = contact or pose_frame
    saved = motion._qs()
    pose, object_tf = flange_in_object(motion, pose_frame, name)
    pre_pose, _object_tf = flange_in_object(motion, pre_frame, name)
    motion.model.apply(saved)
    local, tool_x, tool_y = GRASPS[name]
    record = {
        "pose": pose,
        "pre_pose": pre_pose,
        "tcp": np.eye(4),
        "qpos": np.array([pose_frame["jaw"]], dtype=float),
        "pre_qpos": np.array([pre_frame["jaw"]], dtype=float),
        "score": 1.0,
        "qs": np.asarray(pose_frame["qs"], dtype=float),
        "pre_qs": np.asarray(pre_frame["qs"], dtype=float),
        "object_tf": object_tf,
        "provenance": {
            "object": name,
            "planner": "kitchen_pick",
            "local_contact": np.asarray(local, dtype=float),
            "tool_x": np.asarray(tool_x, dtype=float),
            "tool_y": np.asarray(tool_y, dtype=float),
            "jaw_gap": float(GRASP_GAP[name]),
        },
    }
    GRASP_DIR.mkdir(parents=True, exist_ok=True)
    path = GRASP_DIR / f"{name}.pkl"
    with path.open("wb") as handle:
        pickle.dump(record, handle)
    return path


def draw_grasp(motion, name, record):
    """Object plus the pre-grasp and the closed grasp read out of the pickle."""
    obj = motion.model.objects[name][1]
    pos, rot = record["object_tf"][:3, 3], record["object_tf"][:3, :3]
    obj.set_pos_rotmat(pos=pos, rotmat=rot)
    batches = capture(obj)
    visual = motion._jaw_visual
    gap = float(record["provenance"]["jaw_gap"])
    for joints, jaw, rgb, alpha in (
        (record["pre_qs"], float(record["pre_qpos"][0]), (1.0, 0.93, 0.55), 0.22),
        (record["qs"], float(record["qpos"][0]), (0.72, 0.48, 0.05), 0.82),
    ):
        visual.geom = jaw_mesh(
            motion._jaw_open, motion._jaw_mid, motion._jaw_closed, jaw, gap,
        )
        motion.model.apply(joints)
        for link in motion.model.robot.runtime_lnks:
            batches.extend(capture(link, rgb, alpha))
    return batches


def main():
    parser = argparse.ArgumentParser(description="按物体生成抓取 pickle，并用 Panda3D 查看")
    parser.add_argument("--show", metavar="OBJECT", help="写好后在 Panda3D 里看这个物体的抓取")
    parser.add_argument("--shot", help=argparse.SUPPRESS)
    args = parser.parse_args()
    names = (args.show,) if args.show else tuple(GRASPS)
    unknown = [name for name in names if name not in GRASPS]
    if unknown:
        raise SystemExit(f"没有可用抓取: {', '.join(unknown)}。可选: {', '.join(GRASPS)}")
    motion = kitchen()
    written = {}
    for name in names:
        path = write_grasp(motion, name)
        written[name] = path
        print(path)
    if args.show is None:
        return
    with written[args.show].open("rb") as handle:
        record = pickle.load(handle)
    center = record["object_tf"][:3, 3]
    show(
        draw_grasp(motion, args.show, record),
        f"抓取 · {args.show}",
        look=center,
        eye=center + np.array([1.2, -1.5, 0.85]),
        shot=args.shot,
    )


if __name__ == "__main__":
    main()
