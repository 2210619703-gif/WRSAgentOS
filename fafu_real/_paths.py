"""把本仓库的 fafu_arm_sdk 挂到 sys.path，电机模块用这份，不用比赛仓库里的旧 SDK。

运动学仍要比赛仓库里的 Panthera-HT（``import wrs``）。``demo/real`` 不再加入路径。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 比赛仓库自带的 trimesh 是按 NumPy 1 写的。NumPy 2 删了 np.fromstring 的二进制读法，
# 也不再允许把 0 维数组交给 np.arange。这里只补这两处，不改比赛仓库。
_np_fromstring = np.fromstring
_np_arange = np.arange


def _fromstring_compat(string, dtype=float, count=-1, sep="", **kwargs):
    if sep == "" and isinstance(string, (bytes, bytearray, memoryview)):
        return np.frombuffer(string, dtype=dtype, count=count)
    return _np_fromstring(string, dtype=dtype, count=count, sep=sep, **kwargs)


def _py_scalar(value):
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def _arange_compat(*args, **kwargs):
    args = tuple(_py_scalar(a) for a in args)
    kwargs = {key: _py_scalar(value) for key, value in kwargs.items()}
    return _np_arange(*args, **kwargs)


np.fromstring = _fromstring_compat
np.arange = _arange_compat

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTOS_ROOT = os.path.dirname(HERE)

import fafu_config as C  # noqa: E402

SDK_DIR = os.path.join(AGENTOS_ROOT, "fafu_arm_sdk", "fafu_robot_python")
COMPETITION_ROOT = os.path.abspath(C.COMPETITION_ROOT)
PANTHERA = os.path.join(COMPETITION_ROOT, "wrs", "robot_sim", "robots", "panthera_ht")


def _require(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"找不到{what}：{path}")


_require(os.path.join(SDK_DIR, "fafu_robot_controller.py"), " 本仓库 fafu_arm_sdk 的控制器")
_require(os.path.join(SDK_DIR, "fafu_motor.cp312-win_amd64.pyd"), " Python 3.12 的 fafu_motor")
_require(os.path.join(SDK_DIR, "robot.cfg"), " fafu_arm_sdk 的 robot.cfg")
_require(PANTHERA, "比赛仓库里的 Panthera-HT 运动学模型（只用于规划，不连电机）")

loaded = sys.modules.get("wrs")
if loaded is not None and not os.path.abspath(loaded.__file__).startswith(COMPETITION_ROOT):
    raise SystemExit(f"已经导入了另一份 wrs：{loaded.__file__}；规划要用比赛仓库的 Panthera-HT。")

for path in (AGENTOS_ROOT, COMPETITION_ROOT, SDK_DIR):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
os.environ["FAFU_SDK_DIR"] = SDK_DIR
