import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import warnings


def _load_label_names(path: str) -> List[str]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and "label_names" in obj:
        return [str(x) for x in obj["label_names"]]
    raise ValueError(f"Unrecognized label_names.json format in {path}")


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    *,
    max_tick_labels: int = 50,
) -> None:
    n = cm.shape[0]
    fig_w = min(18, max(6, n * 0.35))
    fig_h = min(18, max(6, n * 0.35))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()

    show_ticks = n <= max_tick_labels
    if show_ticks:
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


def evaluate_and_save(
    *,
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    results_dir: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": f1,
        "num_classes": int(len(label_names)),
        "num_test_samples": int(len(y_test)),
    }

    metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
    cm_npy_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.npy")
    cm_png_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    _save_json(metrics_path, metrics)
    np.save(cm_npy_path, cm)
    _plot_confusion_matrix(cm, label_names, cm_png_path)
    return metrics, cm


def l1_sparsity_report(
    *,
    coef: np.ndarray,
    feature_names: List[str],
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Summarize L1 sparsity from LogisticRegression.coef_.

    coef shape is (n_classes, n_features) for linear models in sklearn.
    """
    nonzero = np.abs(coef) > eps
    nonzero_counts_per_feature = nonzero.sum(axis=0).tolist()
    # Consider a feature "selected" if any class has a non-zero coefficient.
    counts = np.asarray(nonzero_counts_per_feature, dtype=np.int64)
    selected = (counts > 0).tolist()

    selected_feature_names = [fn for fn, s in zip(feature_names, selected) if s]
    return {
        "eps": eps,
        "nonzero_counts_per_feature": nonzero_counts_per_feature,
        "selected_feature_names": selected_feature_names,
        "num_selected_features": int(sum(selected)),
        "feature_dim": int(coef.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on X_features.npy.")
    parser.add_argument("--features-dir", default="data/processed", help="Directory with X/y/label_names files.")
    parser.add_argument("--categories", type=int, default=None, help="Train on first K categories (labels 0..K-1).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--results-dir", default="results", help="Where to write metrics/plots.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    X_path = os.path.join(args.features_dir, "X_features.npy")
    y_path = os.path.join(args.features_dir, "y_features.npy")
    label_path = os.path.join(args.features_dir, "label_names.json")
    config_path = os.path.join(args.features_dir, "feature_config.json")

    X = np.load(X_path)
    y = np.load(y_path)
    label_names_full = _load_label_names(label_path)

    if args.categories is not None:
        k = int(args.categories)
        if k <= 0:
            raise ValueError("--categories must be >= 1")
        mask = y < k
        X = X[mask]
        y = y[mask]
        label_names = label_names_full[:k]
    else:
        label_names = label_names_full

    # Load feature_names if available (for L1 sparsity output).
    feature_names: List[str]
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        feature_names = [str(x) for x in cfg.get("feature_names", [])]
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if len(feature_names) != X.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Split (stratify helps macro-F1 stability)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )

    # Standardize features for linear models
    # Logistic Regression (L2)
    lr_l2 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )

    evaluate_and_save(
        model_name="lr_l2",
        model=lr_l2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    # Logistic Regression (L1) for feature selection
    lr_l1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=5000,
                    n_jobs=-1,
                    tol=1e-3,
                    C=1.0,
                    multi_class="ovr",
                ),
            ),
        ]
    )

    evaluate_and_save(
        model_name="lr_l1",
        model=lr_l1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    # Extract sparsity from trained LR L1
    # (re-fit already happened inside evaluate_and_save, but we don't have access to the inner coef there)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr_l1.fit(X_train, y_train)
    inner_l1 = lr_l1.named_steps["clf"]
    coef = inner_l1.coef_
    sparsity = l1_sparsity_report(coef=coef, feature_names=feature_names)
    _save_json(os.path.join(args.results_dir, "lr_l1_sparsity.json"), sparsity)

    # Linear SVM
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=args.random_seed, max_iter=20000)),
        ]
    )
    evaluate_and_save(
        model_name="svm_linear",
        model=svm,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    print(f"Done. Wrote results to: {args.results_dir}")


if __name__ == "__main__":
    main()

