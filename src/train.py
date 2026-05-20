from __future__ import annotations

import argparse
import copy
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import INDEX_TO_CHANNEL, VideoSelectorDataset, compute_class_weights
from src.model import VideoSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selector-feature network.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path for resume.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_paths(config: dict[str, Any], video_names: list[str]) -> tuple[list[Path], list[Path]]:
    video_dir = Path(config["data"].get("video_dir", ""))
    label_dir = Path(config["data"]["label_dir"])
    video_paths = [video_dir / f"{name}.mp4" for name in video_names]
    label_paths = [label_dir / f"{name}.json" for name in video_names]
    missing = [str(path) for path in label_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing label files:\n" + "\n".join(missing))
    return video_paths, label_paths


def create_datasets(config: dict[str, Any]) -> tuple[VideoSelectorDataset, VideoSelectorDataset]:
    data_cfg = config["data"]
    common_kwargs = {
        "cache_root": data_cfg["cache_dir"],
        "n_frames": int(data_cfg["n_frames"]),
        "margin": float(data_cfg["relabel_margin"]),
        "cache_frames": False,
    }
    train_video_paths, train_label_paths = resolve_paths(config, list(data_cfg["train_videos"]))
    val_video_paths, val_label_paths = resolve_paths(config, list(data_cfg["val_videos"]))

    train_dataset = VideoSelectorDataset(train_video_paths, train_label_paths, **common_kwargs)
    val_dataset = VideoSelectorDataset(val_video_paths, val_label_paths, **common_kwargs)

    feature_mean, feature_std = train_dataset.compute_feature_normalization()
    train_dataset.set_feature_normalization(feature_mean, feature_std)
    val_dataset.set_feature_normalization(feature_mean, feature_std)
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: VideoSelectorDataset,
    val_dataset: VideoSelectorDataset,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    train_cfg = config["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    pin_memory = torch.cuda.is_available()
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def build_model(config: dict[str, Any]) -> VideoSelector:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return VideoSelector(
        n_frames=int(data_cfg["n_frames"]),
        n_classes=int(model_cfg["n_classes"]),
        input_dim=int(model_cfg.get("input_dim", 25)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        aggregation=str(model_cfg["aggregation"]),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )


def build_phase_components(
    model: VideoSelector,
    config: dict[str, Any],
    phase_name: str,
) -> tuple[torch.optim.Optimizer, CosineAnnealingLR]:
    train_cfg = config["train"]
    if phase_name == "phase1":
        lr = float(train_cfg["lr_phase1"])
        epochs = int(train_cfg["phase1_epochs"])
    elif phase_name == "phase2":
        lr = float(train_cfg["lr_classifier_phase2"])
        epochs = int(train_cfg["phase2_epochs"])
    else:
        raise ValueError(f"Unsupported phase: {phase_name}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return optimizer, scheduler


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train_mode: bool,
    phase_name: str,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.train(mode=train_mode)
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(
        dataloader,
        desc=f"{phase_name} epoch {epoch}/{total_epochs} {'train' if train_mode else 'val'}",
        leave=False,
    )
    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            if train_mode and optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    acc = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
    return avg_loss, float(acc), np.asarray(all_targets), np.asarray(all_predictions)


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": accuracy_score(targets, predictions) if len(targets) > 0 else 0.0,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: VideoSelector,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: dict[str, Any],
    epoch: int,
    phase_name: str,
    val_acc: float,
    best_val_acc: float,
    class_weights: Tensor | None,
    feature_names: tuple[str, ...],
    feature_mean: Tensor | None,
    feature_std: Tensor | None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": copy.deepcopy(config),
        "epoch": epoch,
        "phase": phase_name,
        "val_acc": float(val_acc),
        "best_val_acc": float(best_val_acc),
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "feature_names": list(feature_names),
        "feature_mean": feature_mean.detach().cpu().tolist() if feature_mean is not None else None,
        "feature_std": feature_std.detach().cpu().tolist() if feature_std is not None else None,
    }
    torch.save(payload, checkpoint_path)


def log_epoch_summary(
    logger: logging.Logger,
    phase_name: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    per_class_f1: np.ndarray,
    lrs: list[float],
) -> None:
    logger.info(
        "[%s][Epoch %d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.4f",
        phase_name,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        val_acc,
    )
    logger.info(
        "per_class_f1: kalman=%.4f gmc=%.4f inference=%.4f",
        per_class_f1[0],
        per_class_f1[1],
        per_class_f1[2],
    )
    logger.info("lr=%s", ", ".join(f"{lr:.1e}" for lr in lrs))


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("train")

    config = load_config(args.config)
    train_cfg = config["train"]
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    feature_mean = train_dataset.feature_mean
    feature_std = train_dataset.feature_std

    class_weights = None
    if bool(train_cfg["use_class_weights"]):
        class_weights = compute_class_weights(train_dataset.get_labels()).to(device)
        logger.info("Class weights: %s", [f"{value:.4f}" for value in class_weights.tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(config).to(device)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(train_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_acc = 0.0
    resume_payload: dict[str, Any] | None = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_payload["model_state_dict"])
        best_val_acc = float(resume_payload.get("best_val_acc", resume_payload.get("val_acc", 0.0)))
        logger.info(
            "Resumed from %s, phase=%s, epoch=%s, best_val_acc=%.4f",
            args.resume,
            resume_payload.get("phase"),
            resume_payload.get("epoch"),
            best_val_acc,
        )

    phase1_epochs = int(train_cfg["phase1_epochs"])
    phase2_epochs = int(train_cfg["phase2_epochs"])
    early_stopping_patience = train_cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")

    should_stop_early = False
    stale_epochs = 0

    def run_phase(phase_name: str, total_epochs: int, phase_offset: int) -> None:
        nonlocal best_val_acc, should_stop_early, stale_epochs
        if total_epochs <= 0:
            return

        optimizer, scheduler = build_phase_components(model, config, phase_name)
        start_epoch = 1

        if resume_payload is not None and resume_payload.get("phase") == phase_name:
            saved_epoch = int(resume_payload["epoch"])
            if saved_epoch < total_epochs:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
                start_epoch = saved_epoch + 1
                logger.info("%s resumed at epoch %d", phase_name, start_epoch)
            else:
                logger.info("%s already completed; skipping.", phase_name)
                return

        for epoch in range(start_epoch, total_epochs + 1):
            train_loss, train_acc, train_targets, train_predictions = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train_mode=True,
                phase_name=phase_name,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            val_loss, val_acc, val_targets, val_predictions = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train_mode=False,
                phase_name=phase_name,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            scheduler.step()

            train_metrics = compute_metrics(train_targets, train_predictions)
            val_metrics = compute_metrics(val_targets, val_predictions)
            global_epoch = phase_offset + epoch

            writer.add_scalar("loss/train", train_loss, global_step=global_epoch)
            writer.add_scalar("loss/val", val_loss, global_step=global_epoch)
            writer.add_scalar("acc/train", train_acc, global_step=global_epoch)
            writer.add_scalar("acc/val", val_acc, global_step=global_epoch)
            for class_idx, class_name in INDEX_TO_CHANNEL.items():
                writer.add_scalar(
                    f"f1/train_{class_name}",
                    float(train_metrics["f1"][class_idx]),
                    global_step=global_epoch,
                )
                writer.add_scalar(
                    f"f1/val_{class_name}",
                    float(val_metrics["f1"][class_idx]),
                    global_step=global_epoch,
                )
            for group_idx, lr in enumerate(scheduler.get_last_lr()):
                writer.add_scalar(f"lr/group_{group_idx}", lr, global_step=global_epoch)

            log_epoch_summary(
                logger=logger,
                phase_name=phase_name,
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                per_class_f1=val_metrics["f1"],
                lrs=scheduler.get_last_lr(),
            )

            current_best = max(best_val_acc, float(val_acc))
            save_checkpoint(
                checkpoint_path=checkpoint_dir / "last.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                phase_name=phase_name,
                val_acc=val_acc,
                best_val_acc=current_best,
                class_weights=class_weights,
                feature_names=train_dataset.get_feature_names(),
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            if val_acc >= best_val_acc:
                best_val_acc = float(val_acc)
                best_path = checkpoint_dir / "best.pth"
                save_checkpoint(
                    checkpoint_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    phase_name=phase_name,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    class_weights=class_weights,
                    feature_names=train_dataset.get_feature_names(),
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                )
                logger.info("Best model updated: val_acc=%.4f -> %s", best_val_acc, best_path)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if early_stopping_patience is not None:
                    logger.info("Early stopping counter: %d/%d", stale_epochs, early_stopping_patience)
                    if stale_epochs >= early_stopping_patience:
                        should_stop_early = True
                        logger.info(
                            "Validation did not improve for %d epochs; early stopping.",
                            early_stopping_patience,
                        )
                        break

    should_run_phase1 = (
        resume_payload is None
        or (
            resume_payload.get("phase") == "phase1"
            and int(resume_payload.get("epoch", 0)) < phase1_epochs
        )
    )
    if should_run_phase1:
        run_phase("phase1", phase1_epochs, phase_offset=0)
    else:
        logger.info("Skipping phase1 and entering phase2.")

    if not should_stop_early:
        run_phase("phase2", phase2_epochs, phase_offset=phase1_epochs)

    writer.close()
    logger.info("Training complete. best_val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    main()
