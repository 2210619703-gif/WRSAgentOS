# 实机测试

机械臂、夹爪和煮面技能。相机不在这里开。共用开关在上一级 `fafu_config.py`，手眼标定是上一级的 `hand_eye_calib.py`。

用 Python 3.12，在仓库根目录 `F:\WRSAgentOS-main` 下运行。不加 `--execute` 时不打开串口，只规划。真机执行时每一步要在终端回车，`Ctrl+C` 急停。

## 建议顺序

| 步骤 | 命令 | 作用 |
|---|---|---|
| 1 | `F:\Python312\python.exe fafu_real\robot\main.py --mode check` | 不连硬件，把煮面流程和每个技能规划一遍 |
| 2 | `F:\Python312\python.exe fafu_real\robot\read_joints.py --watch` | 手推机械臂，看关节角和 TCP |
| 3 | `F:\Python312\python.exe fafu_real\robot\read_joints.py --save look` | 把当前姿态记成 `look`。`home`、`B`、`C` 同样记 |
| 4 | 改 `kitchen_scene.py` | 把打印的 `"at_deg"` 贴到对应物体 |
| 5 | `F:\Python312\python.exe fafu_real\robot\main.py --mode goto_pose --pose look --execute` | 第一次真机动，低速 |
| 6 | `F:\Python312\python.exe fafu_real\robot\main.py --mode skill --skill pick --args object=strainer --execute` | 单个技能 |
| 7 | `F:\Python312\python.exe fafu_real\robot\main.py --mode sequence --execute` | `TASK_SEQUENCE` 整段 |
| 8 | `F:\Python312\python.exe fafu_real\robot\main.py --mode node --execute` | 接看板 / Agent |

第 3 步之前如果要标定相机，用上一级的 `fafu_real\hand_eye_calib.py`。

## 模式

在 `fafu_config.py` 里设 `RUN_MODE`，或用 `--mode` 覆盖。

| 模式 | 作用 |
|---|---|
| `read_joints` | 读关节角和 TCP |
| `goto_pose` | 走到命名位形，`--pose` |
| `goto_object` | TCP 朝下，停在物体上方，`--object`、`--above` |
| `gripper` | 张开或合上，`--gripper open` |
| `skill` | 一个技能，`--skill`、`--args object=noodles` |
| `sequence` | 按 `TASK_SEQUENCE` 做完 |
| `check` | 强制 dry-run，把序列和每个技能都规划一遍 |
| `node` | 作为机器人节点上线 |
| `calib` | 转去跑上一级的手眼标定 |

`goto_pose`、`goto_object`、`gripper` 默认走技能（有前置条件和碰撞检查）。调试底层运动时加 `--no-skills`。`skill`、`sequence`、`node` 始终走技能。

## 文件

| 文件 | 作用 |
|---|---|
| `main.py` | 入口 |
| `read_joints.py` | 只读关节，`--save` 写入上一级的 `taught_poses.json` |
| `arm.py` | `FafuRobotController`：夹爪用米，路径分段下发 |
| `kinematics.py` | Panthera-HT 正逆解，并按 `robot.cfg` 软限位检查 |
| `motions.py` | 直线、抬高转场、绕点倾斜，每帧做碰撞检查 |
| `skills.py` | 抓取、放置、入容器、倾倒、搅拌等技能 |
| `world.py` | 物体在哪、手里拿着什么 |
| `kitchen_scene.py` | 每个物体的示教关节角 `at_deg` |
| `node.py` | 路由器、机器人节点、Agent 和看板 |

## 技能在真机上做什么

| 技能 | 动作 |
|---|---|
| `observe` | 读一次关节，刷新世界状态 |
| `move_named_pose` | 走到 `fafu_config` 或 `taught_poses.json` 里的命名位形 |
| `move_relative` | TCP 沿世界 XYZ 平移，不超过 10 cm |
| `set_gripper` | 张开或合上。手里有东西时拒绝张开 |
| `pick` | 张爪、靠近、下探、力控合爪、抬起。没夹到就松开退回 |
| `place` | 放到示教位 `B` 或 `C` |
| `place_on` | 放到盘子、灶或桌面 |
| `seat_in` | 放进容器：放得下就放到口内，放不下就架在口上 |
| `pour` | 移到目标上方，倾斜，停一下，回正 |
| `stir` / `scoop` / `shake_off` | 伸进容器搅拌、舀、或抬起抖一抖 |
| `set_control` | 用合爪按灶的按键 |
| `reorient` | 抬高后翻转 |
| `verify` | 只检查状态，不动 |

规划在运动之前做完。某一帧逆解失败或碰到东西，整个技能不下发。碰撞检查覆盖夹爪和手里的物体，不检查手臂连杆。

物体坐标来自 `kitchen_scene.py` 的示教角，不来自相机。东西被碰歪后，程序不知道，要摆回去再记一次 `at_deg`。
