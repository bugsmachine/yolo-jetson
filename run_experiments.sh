#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

MEDIA_ROOT="/home/tan/Desktop/all"
DATA_ROOT="$MEDIA_ROOT/organized_dataset"
CHECKPOINT="checkpoints/best.pth"
MAX_FRAMES=300
LOG_DIR="results/logs"

mkdir -p results "$LOG_DIR"

SEQUENCES=(
  "drone__M0207"
  "drone__M0605"
  "drone__M0606"
  "static__MVI_20011"
  "static__MVI_20012"
  "static__MVI_20032"
  "static__MVI_20034"
  "static__MVI_20051"
  "static__MVI_20052"
  "static__MVI_20065"
  "static__MVI_39031"
)

declare -a SUMMARY=()

for item in "${SEQUENCES[@]}"; do
  category="${item%%__*}"
  seq="${item##*__}"
  dataset="$DATA_ROOT/$category/$seq"
  video="$dataset/$seq.mp4"
  output_video="results/${item}_pipeline_preview.mp4"
  log_file="$LOG_DIR/${item}.log"

  echo "======================================================================"
  echo "Running: $item"
  echo "Dataset: $dataset"
  echo "Video:   $video"
  echo "Output:  $output_video"
  echo "Log:     $log_file"
  echo "======================================================================"

  python3 main.py "$dataset" \
    --media-root "$MEDIA_ROOT" \
    --selector-checkpoint "$CHECKPOINT" \
    --video "$video" \
    --save-video \
    --output-video "$output_video" \
    --max-frames "$MAX_FRAMES" 2>&1 | tee "$log_file"

  status=${PIPESTATUS[0]}
  SUMMARY+=("$item:$status")

  if [ "$status" -eq 0 ]; then
    echo "[OK] $item"
  else
    echo "[FAIL] $item exit=$status"
  fi
  echo
done

echo "======================================================================"
echo "Batch summary"
echo "======================================================================"
for entry in "${SUMMARY[@]}"; do
  item="${entry%%:*}"
  status="${entry##*:}"
  if [ "$status" -eq 0 ]; then
    echo "[OK]   $item"
  else
    echo "[FAIL] $item exit=$status"
  fi
done
