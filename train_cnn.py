from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import QuickDrawCNN


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
        "--conv-channels",
        type=int,
        nargs=2,
        default=(32, 64),
        metavar=("C1", "C2"),
        help="Output channels for the two convolutional layers.",
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


def ensure_nchw(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


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

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }
    return metrics, targets, preds


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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
    model = QuickDrawCNN(
        num_classes=num_classes,
        input_size=input_size,
        conv_channels=tuple(args.conv_channels),
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

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics = {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0}
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if len(split.X_val) > 0:
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        else:
            val_metrics = train_metrics

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
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
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

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

    metrics = {
        "model": "quickdraw_cnn",
        "device": str(device),
        "num_classes": num_classes,
        "num_train_samples": int(len(split.y_train)),
        "num_val_samples": int(len(split.y_val)),
        "num_test_samples": int(len(split.y_test)),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_metrics["accuracy"]),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }

    save_json(os.path.join(args.results_dir, "cnn_metrics.json"), metrics)
    save_json(os.path.join(args.results_dir, "cnn_classification_report.json"), report)
    save_json(
        os.path.join(args.results_dir, "cnn_history.json"),
        {"history": history},
    )
    np.save(os.path.join(args.results_dir, "cnn_confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, label_names, os.path.join(args.results_dir, "cnn_confusion_matrix.png"))
    plot_training_curves(history, os.path.join(args.results_dir, "cnn_training_curves.png"))

    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Results written to: {args.results_dir}")


if __name__ == "__main__":
    main()
