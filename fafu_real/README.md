# fafu_real

fafu（Panthera-HT）的实机和视觉脚本。用 Python 3.12，在仓库根目录 `F:\WRSAgentOS-main` 下运行。

电机走本仓库 `fafu_arm_sdk`（`fafu_motor.cp312-win_amd64.pyd`）。规划用的 Panthera 模型仍在比赛仓库，路径是 `fafu_config.py` 里的 `COMPETITION_ROOT`。

## 目录

| 位置 | 内容 |
|---|---|
| `fafu_config.py` | 两边共用的开关：速度、串口、相机、命名位形、任务序列、权重路径 |
| `_paths.py` | 把本仓库电机 SDK 和比赛仓库的运动学挂到 `sys.path` |
| `camera.py` | D405：848×480、深度 z16、彩色 bgr8、30 帧，深度对齐到彩色 |
| `handeye.py` | 手眼矩阵读写。世界点 = 法兰正运动学 × `T_flange_cam` × 相机点 |
| `hand_eye_calib.py` | 眼在手上标定窗口 |
| `vision/` | 只测相机和识别，不动机械臂。说明见 `vision/README.md` |
| `robot/` | 只测机械臂和技能。说明见 `robot/README.md` |

标定和读关节会互相用到：标定窗口里的关节来自 `robot/`，识别出的相机点要变成世界坐标时用手眼里的矩阵。所以这两类留在外面。

## 手眼标定

先把臂摆到能看见桌面的位置，再运行：

```powershell
F:\Python312\python.exe fafu_real\hand_eye_calib.py
```

`f` 把点云贴到桌面，`wasd` / `qe` 平移，`zxcvbn` 旋转，对准桌沿、锅和盘子后按 `p` 保存。结果在 `handeye_eye_in_hand.json`。
