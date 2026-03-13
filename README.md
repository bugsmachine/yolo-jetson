## 先看这个

你这边先按这个来就行：

- 想看主程序怎么启动，直接跑 `python3 main.py MOT17-10-SDP`
- 想准备训练数据，就先跑 `generate_yolo_gt.py`，把每个序列的 `gt_yolo.txt` 做出来
- 整理好的数据目录发我，我拿到 Jetson Nano 上统一跑

## 怎么启动

先装依赖：

```bash
pip install numpy opencv-python ultralytics
```

如果只是先生成 `gt_yolo.txt`，其实先装 `ultralytics` 也够用。

主程序现在这样启动：

```bash
python3 main.py MOT17-10-SDP
```

现在 `main.py` 默认还是 `TRAIN_MODE=True`，传一个数据集目录进去之后，它会自己去找：

- 根目录下的 `gt_yolo.txt`
- 目录里的 `.mp4`

所以你那边要准备的数据，最少要有视频和 `gt_yolo.txt`。

## 你那边的数据怎么整理

尽量直接照着 `MOT17-10-SDP` 这个目录样子来，省得后面再改。

最少保证这样：

```text
你的序列目录/
├── xxx.mp4
└── gt_yolo.txt
```

如果你那边本来就有别的东西，也可以继续放着，比如：

```text
你的序列目录/
├── xxx.mp4
├── gt_yolo.txt
├── det/
├── gt/
└── img1/
```

总之核心就是每个序列目录里要有：

- 一个视频
- 一个 `gt_yolo.txt`

如果你那边拿到的是标准 MOT 那种只有 `img1/` 图片序列，没有现成视频，就先自己用 `ffmpeg` 合一个 `.mp4`。按现在这套配置，直接做成 `640x640`、`30fps` 就行，我这边主流程现在就是按这个在跑。我之前的mot的数据是ct不知道在哪个网站找的你可以问问他

比如可以这样：

```bash
ffmpeg -framerate 30 -i img1/%06d.jpg -vf "scale=640:640" -c:v libx264 -pix_fmt yuv420p seq.mp4
```

你把这种目录整理好之后直接发我，我这边上机器跑。

## 怎么生成 gt_yolo.txt

`generate_yolo_gt.py` 现在会用 `yolo11x.pt` 逐帧跑视频，然后在每个序列目录下面生成一个 `gt_yolo.txt`。

先看一下脚本里的这个列表，改成你要处理的目录名：

```python
FOLDERS = ['MOT17-10-SDP', 'MOT17-11-FRCNN', 'MOT17-13-SDP']
```

然后把这些目录放到和脚本同一级，目录里放视频：

```text
all/
├── generate_yolo_gt.py
├── seq_a/
│   └── a.mp4
├── seq_b/
│   └── b.mp4
└── seq_c/
    └── c.mp4
```

直接跑：

```bash
python3 generate_yolo_gt.py
```

脚本会自动下载 `yolo11x.pt`，跑完之后会在每个目录下面生成：

```text
<seq>/gt_yolo.txt
```

这个脚本现在只保留 `person` 类别。

## engine 这块

这个项目最后是泡在 Jetson Nano 上跑的，所以 `.engine` 这块要注意一下。

你要在你自己电脑上把 `main.py` 完整跑起来，需要按你自己那边的环境重新做 engine，文件名先保持下面这三个：

- `yolo11n_int8.engine`
- `yolo11m_int8.engine`
- `yolo11m_320_int8.engine`

TensorRT engine 基本就是跟机器和环境绑着的。

如果你现在只是先做 `gt_yolo.txt` 和前面的 LSTM 部分，那先知道这个限制就行。后面你要把 LSTM 模块接到后端里，最好也顺手把 engine 提前打一下包，这个在不同机器上时间会不一样，Jetson 上会慢很多。

## 训练数据先给多少

我觉得第一轮先给我 `15~20` 组会比较稳，少于 `10` 组意义不太大。

场景别全是一个风格，尽量混一点：

- 直道、画面比较稳的
- 相机一直在动的
- 手持拍的
- 人车进出比较频繁的
- 遮挡多一点和少一点的都带一些

你可以先拿现有的几组数据把流程跑通，确认 `gt_yolo.txt` 生成和接口接入没问题。后面再多找一些 MOT 这种数据做 `gt_yolo.txt` 发我，我这边在机器上跑完训练数据，最后再回给你做训练。
