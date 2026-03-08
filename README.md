## all 目录说明

这个目录是从原项目里单独整理出来的可分发版本，便于直接发给同事或单独建仓库。

### 已包含内容

- 核心脚本
  - `main.py`
  - `mot_evaluator.py`
  - `kalman_tracker.py`
  - `train_collector.py`
  - `run_experiments.sh`
  - `generate_yolo_gt.py`
- 样例数据
  - `MOT17-02-DPM/`
  - 包含视频、`gt/gt.txt`、`det/`、`img1/`、`seqinfo.ini`

### 文件来源

- `main.py` 来自 `v3/main.py`
- `run_experiments.sh` 和 `generate_yolo_gt.py` 来自 `v3/`
- `mot_evaluator.py`、`kalman_tracker.py`、`train_collector.py` 来自 `v2/`
  - 原因：`v3/main.py` 会直接导入这 3 个模块，但 `v3/` 目录里原本没有

### 运行前还需要准备

当前仓库里仍然没有模型文件，运行 `main.py` 时需要把下面几个模型放到和 `main.py` 同一级目录：

- `yolo11n_int8.engine`
- `yolo11m_int8.engine`
- `yolo11m_320_int8.engine`

### Python 依赖

- `numpy`
- `opencv-python`
- `ultralytics`

如果继续使用 `.engine` 模型，目标机器还需要具备 TensorRT 运行环境。

### 运行方式

如果直接使用当前目录自带的样例数据：

```bash
python3 main.py MOT17-02-DPM
```

说明：

- 当前 `main.py` 默认 `TRAIN_MODE=True`
- 传入 `MOT17-02-DPM` 后，程序会在该目录下自动寻找视频
- 如果没有 `gt_yolo.txt`，当前代码不会强制退出，会继续使用默认的 `MOT17-02-DPM/gt/gt.txt`

### 其他数据集

如果要跑其他数据集，建议结构至少满足下面其中一种：

- `<dataset_root>/gt_yolo.txt` + 目录内任意 `.mp4`
- `<dataset_root>/main/<seq_name>.mp4`

如果直接用 `run_experiments.sh`，需要把 `MOT17-10-SDP`、`MOT17-11-FRCNN`、`MOT17-13-SDP` 这几个数据目录和本目录放在同一级。

### 用 `generate_yolo_gt.py` 生成标准 GT

这个脚本会用 `YOLO11x` 对视频逐帧推理，并生成当前项目使用的标准 `gt_yolo.txt` 文件。

输出文件格式是 MOT 风格的 9 列文本：

```text
frame_id,track_id,bb_left,bb_top,bb_width,bb_height,not_ignored,class_id,visibility
```

脚本当前写出的每一行是：

```text
frame_id,-1,bb_left,bb_top,bb_width,bb_height,1,1,1.0
```

使用步骤：

1. 安装依赖：

```bash
pip install ultralytics
```

2. 把要处理的数据集文件夹放到和 `generate_yolo_gt.py` 同一级目录。

目录示例：

```text
all/
├── generate_yolo_gt.py
├── MOT17-10-SDP/
│   └── xxx.mp4
├── MOT17-11-FRCNN/
│   └── xxx.mp4
└── MOT17-13-SDP/
    └── xxx.mp4
```

3. 如果要处理的目录名不是这三个，先修改脚本里的 `FOLDERS`：

```python
FOLDERS = ['MOT17-10-SDP', 'MOT17-11-FRCNN', 'MOT17-13-SDP']
```

4. 在当前目录执行：

```bash
python3 generate_yolo_gt.py
```

5. 脚本会自动加载 `yolo11x.pt`，如果本地没有会由 `ultralytics` 自动下载。

6. 生成结果会写到每个数据集目录下的：

```text
<dataset_root>/gt_yolo.txt
```

补充说明：

- 当前 `main.py` 传入数据集目录时，优先读取 `<dataset_root>/gt_yolo.txt`
- 如果你需要传统命名，也可以把生成的 `gt_yolo.txt` 再复制或重命名成 `gt.txt`
- 脚本默认只保留 `person` 类别，也就是 COCO 的 `class 0`
