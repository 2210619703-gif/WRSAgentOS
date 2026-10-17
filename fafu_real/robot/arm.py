"""用本仓库 fafu_arm_sdk 的 FafuRobotController 驱动机械臂。

不经过比赛仓库的 RealArm。EXECUTE_ON_REAL=False 时不打开串口，只更新虚拟关节角。
真机执行要用 Python 3.12，加载 fafu_motor.cp312-win_amd64.pyd。
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

_REAL = str(Path(__file__).resolve().parent.parent)
if _REAL not in sys.path:
    sys.path.insert(0, _REAL)

import _paths  # noqa: F401
import fafu_config as C

_CFG = os.path.join(_paths.SDK_DIR, "robot.cfg")
_LIMITS = re.compile(r"^limits\.(\d+)\s*=\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)", re.M)


def _read_limits(path: str) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """robot.cfg 的限位是度。关节 1–6 和夹爪 7 都换成弧度。"""
    text = open(path, encoding="utf-8").read()
    found = {
        int(m.group(1)): (math.radians(float(m.group(2))), math.radians(float(m.group(3))))
        for m in _LIMITS.finditer(text)
    }
    missing = [i for i in range(1, 8) if i not in found]
    if missing:
        raise RuntimeError(f"{path} 缺少 limits.{missing}")
    return [found[i] for i in range(1, 7)], found[7]


def _speed(value) -> int:
    """配置里的速度档直接当作百分比。10 大约是 18°/s（100% = π rad/s）。"""
    return max(1, min(100, int(value)))


class Stopped(RuntimeError):
    """收到停止请求，在两段路径之间停下了。"""


class Arm:
    def __init__(self, *, execute: bool, logger=print):
        self.execute = bool(execute)
        self.log = logger
        self.limits, self.gripper_limits = _read_limits(_CFG)
        self.hw = None
        self.width = C.JAW_OPEN_M
        self._q = list(C.named_pose_rad(C.DRY_RUN_START_POSE))

    @property
    def joint_limits(self):
        return self.limits

    def _angle(self, width: float) -> float:
        """开口 0 为合拢，JAW_OPEN_M 为张开。限位低的一端是合拢。"""
        t = min(1.0, max(0.0, float(width) / C.JAW_OPEN_M))
        lo, hi = self.gripper_limits
        return lo + (hi - lo) * t

    def connect(self) -> Arm:
        if not self.execute:
            self.log("[arm] dry-run，不打开串口")
            return self
        from fafu_robot_controller import FafuRobotController

        self.hw = FafuRobotController(
            _CFG,
            port=C.SERIAL_PORT,
            has_gripper=True,
            gripper_motor_id=7,
            auto_enable=True,
        )
        self._q = [float(v) for v in self.hw.get_joint_values()]
        return self

    def close(self) -> None:
        if self.hw is not None:
            mode = C.RELEASE_MODE_ON_CLOSE
            self.hw.close_connection(joint_release=mode, gripper_release=mode)
            self.hw = None

    def joints(self) -> list[float]:
        if self.hw is not None:
            self._q = [float(v) for v in self.hw.get_joint_values()]
        return list(self._q)

    def joint_values(self) -> list[float]:
        return self.joints()

    def move(self, path, *, label: str, speed=None, stop=None) -> list[float]:
        """走一条关节路径（弧度）。真机按 PATH_CHUNK 分段下发，段间可以停。"""
        path = [list(map(float, q)) for q in path]
        if not path:
            return self.joints()
        if not self.execute:
            self._q = path[-1]
            self.log(f"[arm] dry-run {label} → 末点")
            return path[-1]
        rate = _speed(speed or C.MOVE_SPEED)
        for start in range(0, len(path), C.PATH_CHUNK):
            if stop is not None and stop.is_set():
                raise Stopped(label)
            chunk = path[start : start + C.PATH_CHUNK]
            if len(chunk) == 1:
                self.hw.move_j(chunk[0], speed=rate, block=True)
            else:
                self.hw.move_jntspace_path(chunk, speed=rate, start_frame_id=0)
        self._q = path[-1]
        return path[-1]

    def open(self, width: float | None = None, *, label: str = "") -> None:
        width = C.JAW_OPEN_M if width is None else float(width)
        if label:
            self.log(f"  {label}")
        if self.hw is not None:
            self.hw.gripper_control(self._angle(width), effort=80)
        self.width = width

    def grasp(
        self, width: float, *, label: str = "", mode: str = "force", force=None, effort=None
    ) -> bool:
        """合爪。force 用力矩阈值停；pinch 合到指定开口，力矩上限压低。dry-run 按抓到算。"""
        if label:
            self.log(f"  {label}")
        if self.hw is None:
            self.width = float(width)
            return True
        if mode == "pinch":
            return self.pinch(width, effort=int(effort if effort is not None else 70))
        result = self.hw.grasp(
            target_angle=self._angle(width),
            force_threshold=int(force if force is not None else 160),
            effort=int(effort if effort is not None else 220),
        )
        self.width = float(width)
        return bool(result.grasped)

    def pinch(self, width: float, *, effort: int) -> bool:
        self.log(f"  夹爪合到 {width * 1000:.0f}mm 就停，力矩上限 {effort}")
        if self.hw is not None:
            self.hw.gripper_control(self._angle(width), effort=int(effort))
        self.width = float(width)
        return True

    def close_jaw(self, *, label: str = "") -> None:
        if label:
            self.log(f"  {label}")
        if self.hw is not None:
            self.hw.gripper_control(self.gripper_limits[0], effort=80)
        self.width = 0.0

    def jaw_width(self) -> float:
        return float(self.width)

    def estop(self) -> None:
        if self.hw is not None:
            self.hw.emergency_stop()

    def describe_limits(self) -> str:
        parts = [
            f"J{i + 1} {math.degrees(lo):.1f}~{math.degrees(hi):.1f}°"
            for i, (lo, hi) in enumerate(self.limits)
        ]
        return "真机软限位：" + "，".join(parts)
