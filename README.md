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
