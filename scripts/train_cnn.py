from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import QuickDrawCNN, QuickDrawDeepCNN, QuickDrawResNet


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LeNet-style CNN on QuickDraw raster images.")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing X_cnn.npy and y_cnn.npy.")
    parser.add_argument("--images", default=None, help="Optional explicit path to X_cnn.npy.")
    parser.add_argument("--labels", default=None, help="Optional explicit path to y_cnn.npy.")
    parser.add_argument(
        "--label-names",
        default="data/processed/label_names.json",
        help="Optional JSON file containing label names.",
    )
    parser.add_argument("--results-dir", default="results", help="Where to save CNN metrics, plots, and checkpoints.")
    parser.add_argument("--categories", type=int, default=None, help="Train on labels 0..K-1 only.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples after filtering.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in the classifier head.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of the classifier head.")
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="plateau",
        help="Learning-rate scheduler to use after each epoch.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="Patience for ReduceLROnPlateau before lowering the learning rate.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="LR decay factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply light sketch-friendly data augmentation during training.",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=12.0,
        help="Maximum absolute rotation used for training augmentation.",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=0.10,
        help="Maximum translation fraction used for training augmentation.",
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        default=0.10,
        help="Uniform scale jitter used for training augmentation.",
    )
    parser.add_argument(
        "--random-erase-prob",
        type=float,
        default=0.10,
        help="Probability of masking a small random patch during training augmentation.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lenet", "deep", "resnet"],
        default="lenet",
        help="CNN architecture to train. Use `deep` or `resnet` for stronger models.",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=(32, 64),
        metavar="C",
        help="Convolution channel sizes. Use 2 values for `lenet` and 3 values for `deep`/`resnet`.",
    )
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction taken from the training split.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count. Set to 0 in some notebook environments if needed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps", None],
        help="Optional manual device override.",
    )
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        help="Use inverse-frequency class weights in cross-entropy loss.",
    )
    parser.add_argument("--early-stopping", type=int, default=4, help="Stop after this many unimproved validation epochs.")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save an extra checkpoint after each epoch.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_label_names(path: str, num_classes: int) -> List[str]:
    if os.path.exists(path):
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            names = [str(x) for x in obj]
            if len(names) >= num_classes:
                return names[:num_classes]
    return [f"label_{i}" for i in range(num_classes)]


def resolve_conv_channels(model_type: str, channels: List[int]) -> Tuple[int, ...]:
    conv_channels = tuple(int(c) for c in channels)
    if any(c <= 0 for c in conv_channels):
        raise ValueError(f"--conv-channels must all be positive, got {conv_channels}.")

    if model_type == "lenet":
        if len(conv_channels) != 2:
            raise ValueError(
                f"--model-type lenet expects exactly 2 channel values, got {conv_channels}."
            )
        return conv_channels

    if model_type == "deep":
        if len(conv_channels) == 2:
            # Friendly fallback: widen the final stage automatically if only two
            # values are provided so older command patterns still work.
            c1, c2 = conv_channels
            return (c1, c2, min(c2 * 2, 512))
        if len(conv_channels) != 3:
            raise ValueError(
                f"--model-type deep expects 3 channel values, got {conv_channels}."
            )
        return conv_channels

    if model_type == "resnet":
        if len(conv_channels) == 2:
            c1, c2 = conv_channels
            return (c1, c2, min(c2 * 2, 512))
        if len(conv_channels) != 3:
            raise ValueError(
                f"--model-type resnet expects 3 channel values, got {conv_channels}."
            )
        return conv_channels

    raise ValueError(f"Unknown model_type={model_type!r}")


def ensure_nchw(images: np.ndarray) -> np.ndarray:
    if images.ndim == 2:
        side = int(math.isqrt(images.shape[1]))
        if side * side != images.shape[1]:
            raise ValueError(
                "Expected flattened square images for 2D input arrays; "
                f"got shape {images.shape}."
            )
        images = images.reshape(images.shape[0], 1, side, side)
    elif images.ndim == 3:
        images = images[:, None, :, :]
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = np.transpose(images, (0, 3, 1, 2))
    elif images.ndim == 4 and images.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "Expected image array shaped (N,H,W), (N,H,W,1), or (N,1,H,W); "
            f"got {images.shape}."
        )
    return images.astype(np.float32, copy=False)


def normalize_images(images: np.ndarray) -> np.ndarray:
    max_value = float(images.max()) if images.size > 0 else 1.0
    if max_value > 1.0:
        images = images / 255.0
    return np.clip(images, 0.0, 1.0)


def subset_data(X: np.ndarray, y: np.ndarray, categories: Optional[int], limit: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if categories is not None:
        if categories <= 0:
            raise ValueError("--categories must be >= 1")
        mask = y < categories
        X = X[mask]
        y = y[mask]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be >= 1")
        X = X[:limit]
        y = y[:limit]

    if len(X) == 0:
        raise ValueError("No samples remain after filtering. Check --categories/--limit.")
    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> SplitData:
    if not (0.0 < test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("--val-size must be in [0, 1).")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    if val_size == 0.0:
        return SplitData(X_train_val, y_train_val, X_test[:0], y_test[:0], X_test, y_test)

    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=random_seed,
        stratify=y_train_val,
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def build_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64, copy=False)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_topk_accuracies(logits: Tensor, targets: Tensor, ks: Tuple[int, ...] = (3, 5)) -> Dict[str, float]:
    """Compute top-k accuracies from raw logits for the requested k values."""
    num_classes = logits.size(1)
    max_k = min(max(ks), num_classes)
    topk = logits.topk(max_k, dim=1).indices
    targets = targets.view(-1, 1)
    metrics: Dict[str, float] = {}
    for k in ks:
        k_eff = min(k, num_classes)
        correct = (topk[:, :k_eff] == targets).any(dim=1).float().mean().item()
        metrics[f"top_{k}_accuracy"] = float(correct)
    return metrics


def apply_batch_augmentation(
    xb: Tensor,
    *,
    max_rotation_deg: float,
    max_translation: float,
    scale_jitter: float,
    random_erase_prob: float,
) -> Tensor:
    """
    Apply light geometric perturbations that preserve sketch semantics.

    We keep augmentation intentionally mild because aggressive transforms can
    distort small 28x28 doodles more than they help.
    """
    batch_size = xb.size(0)
    device = xb.device
    dtype = xb.dtype

    angles = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        -max_rotation_deg, max_rotation_deg
    ) * (math.pi / 180.0)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    scale_low = max(0.7, 1.0 - scale_jitter)
    scale_high = 1.0 + scale_jitter
    scales = torch.empty(batch_size, device=device, dtype=dtype).uniform_(scale_low, scale_high)
    tx = torch.empty(batch_size, device=device, dtype=dtype).uniform_(-max_translation, max_translation)
    ty = torch.empty(batch_size, device=device, dtype=dtype).uniform_(-max_translation, max_translation)

    theta = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a / scales
    theta[:, 0, 1] = -sin_a / scales
    theta[:, 1, 0] = sin_a / scales
    theta[:, 1, 1] = cos_a / scales
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, xb.size(), align_corners=False)
    xb = F.grid_sample(xb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if random_erase_prob > 0:
        erase_mask = torch.rand(batch_size, device=device) < random_erase_prob
        _, _, height, width = xb.shape
        for idx in torch.nonzero(erase_mask, as_tuple=False).flatten().tolist():
            erase_h = max(2, int(height * float(torch.empty(1).uniform_(0.12, 0.25).item())))
            erase_w = max(2, int(width * float(torch.empty(1).uniform_(0.12, 0.25).item())))
            top = int(torch.randint(0, max(height - erase_h + 1, 1), (1,), device=device).item())
            left = int(torch.randint(0, max(width - erase_w + 1, 1), (1,), device=device).item())
            xb[idx, :, top:top + erase_h, left:left + erase_w] = 0.0

    return xb.clamp_(0.0, 1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    augment: bool,
    max_rotation_deg: float,
    max_translation: float,
    scale_jitter: float,
    random_erase_prob: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_logits: List[Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if augment:
            xb = apply_batch_augmentation(
                xb,
                max_rotation_deg=max_rotation_deg,
                max_translation=max_translation,
                scale_jitter=scale_jitter,
                random_erase_prob=random_erase_prob,
            )

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    logits_cpu = torch.cat(all_logits, dim=0)
    topk_metrics = compute_topk_accuracies(
        logits_cpu,
        torch.from_numpy(targets.astype(np.int64, copy=False)),
    )
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }
    metrics.update(topk_metrics)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_logits: List[Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(yb.cpu().numpy())
        all_logits.append(logits.cpu())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    logits_cpu = torch.cat(all_logits, dim=0)
    topk_metrics = compute_topk_accuracies(
        logits_cpu,
        torch.from_numpy(targets.astype(np.int64, copy=False)),
    )
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }
    metrics.update(topk_metrics)
    return metrics, targets, preds


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_top_confusions(cm: np.ndarray, label_names: List[str], *, top_n: int = 10) -> List[Dict[str, Any]]:
    """Return the most common off-diagonal confusion pairs for presentation use."""
    confusions: List[Dict[str, Any]] = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            confusions.append(
                {
                    "true_label": label_names[true_idx],
                    "predicted_label": label_names[pred_idx],
                    "count": count,
                }
            )
    confusions.sort(key=lambda row: row["count"], reverse=True)
    return confusions[:top_n]


def build_worst_classes(report: Dict[str, Any], label_names: List[str], *, top_n: int = 10) -> List[Dict[str, Any]]:
    """Return the weakest per-class metrics from the classification report."""
    rows: List[Dict[str, Any]] = []
    for label_name in label_names:
        stats = report.get(label_name)
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "label": label_name,
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1_score": float(stats.get("f1-score", 0.0)),
                "support": int(stats.get("support", 0)),
            }
        )
    rows.sort(key=lambda row: (row["f1_score"], row["recall"], row["precision"]))
    return rows[:top_n]


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, *, max_tick_labels: int = 50) -> None:
    n = cm.shape[0]
    fig_w = min(18, max(6, n * 0.35))
    fig_h = min(18, max(6, n * 0.35))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.colorbar()

    if n <= max_tick_labels:
        plt.xticks(range(n), labels, rotation=90, fontsize=6)
        plt.yticks(range(n), labels, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_training_curves(history: List[Dict[str, float]], out_path: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_checkpoint(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    input_size: int,
    best_epoch: int,
    best_val_metrics: Dict[str, float],
    label_names: List[str],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": args.model_type,
            "num_classes": len(label_names),
            "input_size": input_size,
            "conv_channels": tuple(args.conv_channels),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "label_names": label_names,
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    os.makedirs(args.results_dir, exist_ok=True)

    images_path = args.images or os.path.join(args.data_dir, "X_cnn.npy")
    labels_path = args.labels or os.path.join(args.data_dir, "y_cnn.npy")

    X = np.load(images_path)
    y = np.load(labels_path).astype(np.int64)
    X = ensure_nchw(X)
    X = normalize_images(X)
    X, y = subset_data(X, y, args.categories, args.limit)

    input_size = int(X.shape[-1])
    if X.shape[-2] != input_size:
        raise ValueError(f"Expected square inputs, got {X.shape[-2:]}.")

    split = stratified_split(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.random_seed,
    )

    num_classes = int(max(y)) + 1 if len(y) > 0 else 0
    if args.categories is not None:
        num_classes = args.categories

    label_names = load_label_names(args.label_names, num_classes)
    conv_channels = resolve_conv_channels(args.model_type, args.conv_channels)

    train_loader = build_loader(
        split.X_train,
        split.y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        split.X_val,
        split.y_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        split.X_test,
        split.y_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    if args.model_type == "deep":
        model = QuickDrawDeepCNN(
            num_classes=num_classes,
            input_size=input_size,
            conv_channels=conv_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "resnet":
        model = QuickDrawResNet(
            num_classes=num_classes,
            input_size=input_size,
            conv_channels=conv_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        model = QuickDrawCNN(
            num_classes=num_classes,
            input_size=input_size,
            conv_channels=conv_channels,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)

    class_weights = None
    if args.class_weighting:
        class_weights = compute_class_weights(split.y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "plateau":
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        plateau_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
            )
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        plateau_scheduler = None
    else:
        scheduler = None
        plateau_scheduler = None

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics = {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0}
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            augment=args.augment,
            max_rotation_deg=args.max_rotation_deg,
            max_translation=args.max_translation,
            scale_jitter=args.scale_jitter,
            random_erase_prob=args.random_erase_prob,
        )

        if len(split.X_val) > 0:
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        else:
            val_metrics = train_metrics

        history_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_top_3_accuracy": train_metrics["top_3_accuracy"],
            "train_top_5_accuracy": train_metrics["top_5_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top_3_accuracy": val_metrics["top_3_accuracy"],
            "val_top_5_accuracy": val_metrics["top_5_accuracy"],
        }
        history.append(history_row)

        improved = (
            val_metrics["macro_f1"] > best_val_metrics["macro_f1"]
            or (
                val_metrics["macro_f1"] == best_val_metrics["macro_f1"]
                and val_metrics["accuracy"] > best_val_metrics["accuracy"]
            )
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = make_checkpoint(
                model=model,
                args=args,
                input_size=input_size,
                best_epoch=best_epoch,
                best_val_metrics=best_val_metrics,
                label_names=label_names,
            )
            torch.save(checkpoint, os.path.join(args.results_dir, "cnn_best.pt"))
        else:
            epochs_without_improvement += 1

        if args.save_every_epoch:
            torch.save(
                make_checkpoint(
                    model=model,
                    args=args,
                    input_size=input_size,
                    best_epoch=best_epoch,
                    best_val_metrics=best_val_metrics,
                    label_names=label_names,
                ),
                os.path.join(args.results_dir, f"cnn_epoch_{epoch:03d}.pt"),
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_top3={val_metrics['top_3_accuracy']:.4f} val_top5={val_metrics['top_5_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["macro_f1"])
        elif scheduler is not None:
            scheduler.step()

        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"Early stopping after {epoch} epochs without validation improvement.")
            break

    model.load_state_dict(best_state)
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    labels_range = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    top_confusions = build_top_confusions(cm, label_names)
    worst_classes = build_worst_classes(report, label_names)

    metrics = {
        "model": "quickdraw_cnn",
        "model_type": args.model_type,
        "device": str(device),
        "num_classes": num_classes,
        "num_train_samples": int(len(split.y_train)),
        "num_val_samples": int(len(split.y_val)),
        "num_test_samples": int(len(split.y_test)),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_metrics["accuracy"]),
        "best_val_top_3_accuracy": float(best_val_metrics["top_3_accuracy"]),
        "best_val_top_5_accuracy": float(best_val_metrics["top_5_accuracy"]),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_top_3_accuracy": float(test_metrics["top_3_accuracy"]),
        "test_top_5_accuracy": float(test_metrics["top_5_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "conv_channels": list(conv_channels),
            "scheduler": args.scheduler,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor,
            "augment": args.augment,
            "max_rotation_deg": args.max_rotation_deg,
            "max_translation": args.max_translation,
            "scale_jitter": args.scale_jitter,
            "random_erase_prob": args.random_erase_prob,
            "class_weighting": args.class_weighting,
            "random_seed": args.random_seed,
        },
    }
    presentation_summary = {
        "headline_metrics": {
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_top_3_accuracy": float(test_metrics["top_3_accuracy"]),
            "test_top_5_accuracy": float(test_metrics["top_5_accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "best_val_accuracy": float(best_val_metrics["accuracy"]),
            "best_val_top_3_accuracy": float(best_val_metrics["top_3_accuracy"]),
            "best_val_top_5_accuracy": float(best_val_metrics["top_5_accuracy"]),
            "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
            "best_epoch": int(best_epoch),
        },
        "experiment_setup": metrics["training_config"],
        "dataset_split": {
            "num_classes": num_classes,
            "num_train_samples": int(len(split.y_train)),
            "num_val_samples": int(len(split.y_val)),
            "num_test_samples": int(len(split.y_test)),
        },
        "worst_classes_by_f1": worst_classes,
        "top_confusions": top_confusions,
    }

    save_json(os.path.join(args.results_dir, "cnn_metrics.json"), metrics)
    save_json(os.path.join(args.results_dir, "cnn_classification_report.json"), report)
    save_json(os.path.join(args.results_dir, "cnn_presentation_summary.json"), presentation_summary)
    save_json(
        os.path.join(args.results_dir, "cnn_history.json"),
        {"history": history},
    )
    np.save(os.path.join(args.results_dir, "cnn_confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, label_names, os.path.join(args.results_dir, "cnn_confusion_matrix.png"))
    plot_training_curves(history, os.path.join(args.results_dir, "cnn_training_curves.png"))

    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test top-3 accuracy: {test_metrics['top_3_accuracy']:.4f}")
    print(f"Test top-5 accuracy: {test_metrics['top_5_accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Results written to: {args.results_dir}")


if __name__ == "__main__":
    main()
