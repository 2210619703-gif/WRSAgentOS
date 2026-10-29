# 视觉测试

只开 D405，不连机械臂、不下发运动。相机参数在上一级的 `fafu_config.py`（`CAMERA_RESOLUTION`、`CAMERA_SERIAL`）。

相机插在 USB3 口上。命令都在仓库根目录执行。

## 看画面

彩色和深度并排，不加载权重。`q` 或 Esc 关闭。

```powershell
F:\Python312\python.exe fafu_real\vision\watch.py --live
```

## 拍一张

把当前帧存成彩色图、深度和内参，放在 `dataset/outputs/frame`。

```powershell
F:\Python312\python.exe fafu_real\vision\vision.py
```

已经有 `dataset/weights/best.pt` 时，加上 `--yolo` 会在图上检测，并打印每个框中心的相机坐标。

## 自己训练

拍照、拉框、训练的步骤在 `dataset/README.md`。训练完成后：

```powershell
F:\Python312\python.exe fafu_real\vision\watch.py --yolo
```

绿色框过了置信度，橙色是分数不够。`[` `]` 调阈值，`-` `=` 调输入尺寸，`s` 存图。

## 这个文件夹里的文件

| 文件 | 作用 |
|---|---|
| `watch.py` | 实时窗口。`--live` 只看彩色和深度，`--yolo` 用训练好的 `best.pt` |
| `vision.py` | 拍一帧，可选检测 |
| `locate.py` | 真机 `--yolo` 时：检测框经手眼变成世界坐标，覆盖看见的物体 |
| `perception.py` | 把相机、YOLO-World、FastSAM 接到看板用的路由器。令牌要和当时的 `robot/main.py --mode node` 一致 |
| `dataset/` | 拍照、标注、训练 |

真机上加 `--yolo` 时，看见的物体优先用这里的识别位置，没看见的仍用 `robot/kitchen_scene.py` 的示教关节角。示教文件里不写识别结果。
