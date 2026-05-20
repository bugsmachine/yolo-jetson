# src README

This directory contains the training, evaluation, and inference code for the
channel selector model. The selector does not run YOLO and does not read raw
images directly. It learns from structured per-frame features stored in the
label JSON files.

## Goal

For each target frame, the model chooses one of three tracking/detection
channels:

| Class id | Channel | Meaning |
| --- | --- | --- |
| 0 | `kalman` | Use Kalman prediction only |
| 1 | `gmc` | Use global motion compensation prediction |
| 2 | `inference` | Run GPU inference / YOLO correction |

The goal is to keep tracking quality high while avoiding unnecessary GPU
inference.

## Current Model

The main model is `VideoSelector` in `model.py`.

With the default config:

```yaml
data:
  n_frames: 30

model:
  aggregation: gru
  input_dim: 23
  hidden_dim: 128
  dropout: 0.3
  n_classes: 3
```

The default network is a GRU-based classifier:

```text
input: [B, 30, 23]
  -> Linear(23 -> 128)
  -> LayerNorm
  -> ReLU
  -> Dropout(0.3)
  -> GRU(128 -> 128)
  -> last hidden state: [B, 128]
  -> Linear(128 -> 128)
  -> ReLU
  -> Dropout(0.3)
  -> Linear(128 -> 3)
output: logits [B, 3]
```

So the model is not an image model. It is a lightweight temporal classifier
over a window of selector features.

Supported aggregation modes:

```text
mean | concat | diff | lstm | gru
```

Change `model.aggregation` in `configs/default.yaml` to switch modes. The
current default is `gru`.

## Input

Each training sample is a sequence:

```text
x.shape = [n_frames, input_dim]
default = [30, 23]
```

During batching:

```text
inputs.shape = [B, 30, 23]
```

The 23 features come from each frame's JSON field:

```json
"selector_features": {
  ...
}
```

Feature order is defined by `SELECTOR_FEATURE_NAMES` in `dataset.py`:

```text
frame_diff_mean
frame_diff_std
global_motion_dx
global_motion_dy
optical_flow_valid_ratio
optical_flow_residual
tracked_point_count
is_bootstrap_frame
last_gpu_box_count
last_gpu_max_conf
recent_gpu_box_count_mean
recent_gpu_box_count_delta
tracker_count
confirmed_tracker_count
mean_track_age
mean_time_since_update
max_time_since_update
mean_speed
mean_position_uncertainty
max_position_uncertainty
prediction_error_ma
prediction_error_p95
frames_since_last_gpu
```

If a sample does not have enough previous frames, the dataset pads the history
by repeating the earliest available feature vector.

Feature normalization is computed from the training split and then reused for
validation and inference checkpoints.

## Output

The model returns logits:

```text
logits.shape = [B, 3]
```

The class mapping is:

```python
{
    "kalman": 0,
    "gmc": 1,
    "inference": 2,
}
```

Inference code applies softmax to convert logits into probabilities, then picks
the class with the largest probability.

## Labels

Labels are generated from per-frame channel F1 scores in the JSON files.

For each frame, the dataset reads:

```text
channels.kalman.metrics.f1
channels.gmc.metrics.f1
channels.inference.metrics.f1
```

Then `relabel_with_margin()` chooses the label. It prefers cheaper channels
when their F1 is close enough to the best channel:

```text
kalman -> cheapest
gmc -> medium cost
inference -> most expensive
```

The margin is controlled by:

```yaml
data:
  relabel_margin: 0.03
```

Example: if GPU inference has the best F1, but Kalman is within `0.03`, the
label becomes `kalman`. This makes the model cost-aware instead of purely
accuracy-maximizing.

## Main Files

| File | Role |
| --- | --- |
| `dataset.py` | Loads JSON labels, builds feature sequences, creates labels |
| `model.py` | Defines `VideoSelector` |
| `train.py` | Trains the selector and saves checkpoints |
| `evaluate.py` | Evaluates classification and scheduler cost metrics |
| `infer.py` | Runs selector inference and saves per-frame predictions |
| `preprocess.py` | Checks/builds dataset samples and prints label distribution |

## Typical Commands

Run preprocessing / dataset check:

```bash
python -m src.preprocess --config configs/default.yaml
```

Train:

```bash
python -m src.train --config configs/default.yaml
```

Evaluate:

```bash
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pth
```

Run inference:

```bash
python -m src.infer --config configs/default.yaml --checkpoint checkpoints/best.pth --split val
```

## Notes

- `frame_id` and `last_gpu_frame_id` are intentionally excluded from the model
  input because they are absolute frame-index features and can encourage
  overfitting to video position. `frames_since_last_gpu` is kept because it is a
  more transferable relative-state feature.
- For Jetson deployment, GRU is a good default because it is smaller and faster
  than LSTM for this short 30-frame sequence.
- If longer temporal memory becomes important, try `aggregation: lstm` and
  compare validation macro-F1, inference recall, and compute cost.
