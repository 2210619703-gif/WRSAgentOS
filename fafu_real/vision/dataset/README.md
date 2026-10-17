# 厨房检测：拍照、标注、训练

用 D405 自己拍图，标出漏勺、食物和餐具，训练一份小的 YOLOv8n 权重。
这份权重大约 6MB，预览时不用加载 CLIP。

类别写在 `data.yaml`，默认是：

| 序号 | 名字 | 含义 |
|---|---|---|
| 0 | strainer | 漏勺 |
| 1 | food | 食物 |
| 2 | bowl | 碗 |
| 3 | pot | 锅 |
| 4 | spoon | 勺子 |
| 5 | plate | 盘子 |

要改类别，先改 `data.yaml`，再重新标注。已经写好的标签序号不会自动跟着变。

所有命令在仓库根目录 `F:\WRSAgentOS-main` 下，用 Python 3.12：

```powershell
F:\Python312\python.exe fafu_real\vision\dataset\capture.py
```

## 1. 只看相机

不加载权重。窗口左边是彩色，右边是深度。`q` 或 Esc 关闭。

```powershell
F:\Python312\python.exe fafu_real\vision\watch.py --live
```

相机插在 USB3 口上。打不开时先确认没有别的程序占用 D405。

## 2. 拍照

把要认的东西放在相机前面，画面清楚时按 `s`。每按一次保存一张到 `images/train/0001.jpg` 这样的文件名。`q` 退出。

```powershell
F:\Python312\python.exe fafu_real\vision\dataset\capture.py
```

每类至少拍几十张，换一下角度、远近和遮挡。从里面拿出大约两成，剪到 `images/val/`，用来看训练有没有记住训练集。

## 3. 拉框

```powershell
F:\Python312\python.exe fafu_real\vision\dataset\capture.py --label
```

- 鼠标左键拖出一个框
- 数字键 `0` 到 `5` 选择当前类别，再画下一个框
- `z` 撤销最后一个框
- `s` 保存这张图的标签并切到下一张
- `q` 退出

标签写成 `labels/train/0001.txt`，一行一个框：`类别 中心x 中心y 宽 高`，都是相对整张图的 0 到 1。

## 4. 训练

```powershell
F:\Python312\python.exe fafu_real\vision\dataset\train.py
```

默认 50 轮，输入边长 640，用显卡 `0`。第一次会下载基础模型 `yolov8n.pt`。
训练完把最好的一轮复制到 `weights/best.pt`。

轮数可以改：

```powershell
F:\Python312\python.exe fafu_real\vision\dataset\train.py --epochs 80 --device 0
```

`images/val` 还是空的时候，这次会用训练集充当验证，屏幕上会提示。那个分数不能当真。

## 5. 用训练好的权重看画面

```powershell
F:\Python312\python.exe fafu_real\vision\watch.py --yolo
```

绿色框是过了置信度的，橙色是看见了但分数不够。`[` `]` 调阈值，`-` `=` 调输入尺寸，`s` 把当前画面存到 `dataset/outputs`。

只拍一张并打印框中心的相机坐标：

```powershell
F:\Python312\python.exe fafu_real\vision\vision.py --yolo
```

识别结果还没有接到抓取技能。机械臂去哪个位置，仍然用 `fafu_real\robot\kitchen_scene.py` 里的示教关节角。
