#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import main as pipeline


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the channel-selector pipeline and save a full boxed visualization video."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Input video path. If omitted, videos under --media-root are listed for selection.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults to results/visualized_<video_stem>.mp4.",
    )
    parser.add_argument(
        "--media-root",
        default=pipeline.Config.MEDIA_ROOT,
        help="Root containing videos and TensorRT engines.",
    )
    parser.add_argument(
        "--selector-checkpoint",
        default=pipeline.Config.SELECTOR_CHECKPOINT,
        help="Runtime selector checkpoint path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug cap. Default 0 means render the whole video.",
    )
    parser.add_argument(
        "--simulate-yolo",
        action="store_true",
        help="Skip TensorRT engine loading and use simulated detections for local visualization tests.",
    )
    parser.add_argument(
        "--singleprocess",
        action="store_true",
        help="Use the single-process path instead of the shared-memory multiprocess path.",
    )
    parser.add_argument(
        "--channel",
        choices=("auto", "kalman", "gmc"),
        default="auto",
        help="Scheduler mode for visualization. auto uses the selector; kalman/gmc force continuous prediction.",
    )
    parser.add_argument(
        "--yolo-interval",
        type=int,
        default=30,
        help="When --channel is kalman or gmc, force one YOLO correction every N frames. Use 0 to disable.",
    )
    return parser.parse_args()


def find_videos(media_root: Path) -> list[Path]:
    if not media_root.exists():
        return []
    return sorted(
        path
        for path in media_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def select_video(media_root: Path) -> Path:
    videos = find_videos(media_root)
    if not videos:
        raise FileNotFoundError(f"No videos found under {media_root}")

    print(f"Found {len(videos)} videos under {media_root}:")
    for idx, path in enumerate(videos[:100], start=1):
        print(f"{idx:3d}. {path}")
    if len(videos) > 100:
        print(f"... showing first 100 only; pass --video for another file.")

    while True:
        raw = input("Select video number: ").strip()
        try:
            selected_idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= selected_idx <= min(len(videos), 100):
            return videos[selected_idx - 1]
        print("Selection out of range.")


def configure_pipeline(args: argparse.Namespace, video_path: Path, output_path: Path) -> None:
    media_root = Path(args.media_root)

    pipeline.Config.MEDIA_ROOT = str(media_root)
    pipeline.Config.VIDEO_PATH = str(video_path)
    pipeline.Config.VIS_VIDEO_PATH = str(output_path)
    pipeline.Config.SAVE_VIS_VIDEO = True
    pipeline.Config.DISPLAY_OUTPUT = False
    pipeline.Config.TRAIN_MODE = False
    pipeline.Config.ENABLE_MOT_EVAL = False
    pipeline.Config.TEST_MAX_FRAMES = max(0, int(args.max_frames))
    pipeline.Config.SIMULATE_YOLO = bool(args.simulate_yolo)
    pipeline.Config.ENABLE_MULTIPROCESSING = not bool(args.singleprocess)
    pipeline.Config.SELECTOR_CHECKPOINT = str(args.selector_checkpoint)
    pipeline.Config.FORCE_VIS_CHANNEL = str(args.channel)
    pipeline.Config.FORCE_VIS_YOLO_INTERVAL = (
        max(0, int(args.yolo_interval)) if args.channel != "auto" else 0
    )

    pipeline.Config.MODEL_YOLO11N = str(media_root / "yolo11n_int8.engine")
    pipeline.Config.MODEL_YOLO11M_FULL = str(media_root / "yolo11m_int8.engine")
    pipeline.Config.MODEL_YOLO11M_ROI = str(media_root / "yolo11m_320_int8.engine")


def main() -> None:
    args = parse_args()
    media_root = Path(args.media_root)
    video_path = Path(args.video) if args.video else select_video(media_root)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"visualized_{video_path.stem}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_pipeline(args, video_path, output_path)

    print(f"Input video:  {video_path}")
    print(f"Output video: {output_path}")
    print("Box colors: Kalman=blue, GMC=orange, YOLO=green")
    if pipeline.Config.FORCE_VIS_CHANNEL == "auto":
        print("Scheduler: auto selector")
    else:
        print(
            f"Scheduler: continuous {pipeline.Config.FORCE_VIS_CHANNEL}, "
            f"YOLO every {pipeline.Config.FORCE_VIS_YOLO_INTERVAL} frames"
        )
    if pipeline.Config.TEST_MAX_FRAMES == 0:
        print("Rendering full video.")
    else:
        print(f"Rendering first {pipeline.Config.TEST_MAX_FRAMES} frames.")

    pipeline.main()


if __name__ == "__main__":
    main()
