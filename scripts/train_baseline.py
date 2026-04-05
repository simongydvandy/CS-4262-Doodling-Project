import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ── helpers ────────────────────────────────────────────────────────────────────

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


# ── core evaluator ─────────────────────────────────────────────────────────────

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
) -> Tuple[Dict[str, Any], np.ndarray, Any]:    # FIX 2: added return type annotation
    # FIX 2: time each model
    print(f"\nTraining {model_name}...")
    t0 = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed/60:.1f} min")

    # FIX 3: check convergence using isinstance instead of hasattr
    clf = model.named_steps["clf"]
    if hasattr(clf, "n_iter_"):
        iters = clf.n_iter_
        max_i = clf.max_iter
        if isinstance(iters, np.ndarray):
            converged = int(iters.max()) < max_i    # saga: check worst class
        else:
            converged = int(iters) < max_i          # lbfgs: single value
        status = "converged" if converged else "DID NOT CONVERGE — consider increasing max_iter"
        print(f"  Iterations: {iters} / {max_i} — {status}")

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1  = float(f1_score(y_test, y_pred, average="macro"))
    cm  = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")

    metrics = {
        "model":            model_name,
        "accuracy":         acc,
        "macro_f1":         f1,
        "num_classes":      int(len(label_names)),
        "num_test_samples": int(len(y_test)),
        "train_time_sec":   round(elapsed, 2),
    }

    metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
    cm_npy_path  = os.path.join(results_dir, f"{model_name}_confusion_matrix.npy")
    cm_png_path  = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")

    _save_json(metrics_path, metrics)
    np.save(cm_npy_path, cm)
    _plot_confusion_matrix(cm, label_names, cm_png_path)

    # FIX 4: save per-class F1 report
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    _save_json(
        os.path.join(results_dir, f"{model_name}_per_class.json"),
        report,
    )

    # FIX 1: return the fitted model so caller can use it without retraining
    return metrics, cm, model


# ── l1 sparsity reporter ───────────────────────────────────────────────────────

def l1_sparsity_report(
    *,
    coef: np.ndarray,
    feature_names: List[str],
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Summarize L1 sparsity. coef shape: (n_classes, n_features).
    Covers all 26 features from the updated extract_features.py.
    A feature is considered selected if any class has a nonzero coefficient for it.
    """
    nonzero = np.abs(coef) > eps
    nonzero_counts_per_feature = nonzero.sum(axis=0).tolist()
    counts   = np.asarray(nonzero_counts_per_feature, dtype=np.int64)
    selected = (counts > 0).tolist()
    selected_feature_names = [fn for fn, s in zip(feature_names, selected) if s]
    return {
        "eps": eps,
        "nonzero_counts_per_feature": nonzero_counts_per_feature,
        "selected_feature_names":     selected_feature_names,
        "num_selected_features":      int(sum(selected)),
        "feature_dim":                int(coef.shape[1]),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline models on X_features.npy."
    )
    parser.add_argument(
        "--features-dir",
        default="data/processed",
        # FIX 8: document the Drive path requirement for Colab
        help=(
            "Directory with X_features.npy, y_features.npy, label_names.json, "
            "feature_config.json. In Colab use the full Drive path, e.g. "
            "/content/drive/MyDrive/quickdraw-classifier/data/processed"
        ),
    )
    parser.add_argument(
        "--categories",
        type=int,
        default=None,
        help="Train on first K categories only (labels 0..K-1). Useful for quick testing.",
    )
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--random-seed", type=int,   default=42)
    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Where to write metrics, confusion matrices, and sparsity reports. "
            "In Colab use the full Drive path, e.g. "
            "/content/drive/MyDrive/quickdraw-classifier/results"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    X_path      = os.path.join(args.features_dir, "X_features.npy")
    y_path      = os.path.join(args.features_dir, "y_features.npy")
    label_path  = os.path.join(args.features_dir, "label_names.json")
    config_path = os.path.join(args.features_dir, "feature_config.json")

    print("Loading data...")
    X = np.load(X_path)
    y = np.load(y_path)
    label_names_full = _load_label_names(label_path)
    print(f"  X: {X.shape}   y: {y.shape}   classes: {len(label_names_full)}")

    # optional category subset
    if args.categories is not None:
        k = int(args.categories)
        if k <= 0:
            raise ValueError("--categories must be >= 1")
        mask = y < k
        X    = X[mask]
        y    = y[mask]
        label_names = label_names_full[:k]
        print(f"  Subset to first {k} categories: {X.shape}")
    else:
        label_names = label_names_full

    # load feature names from config
    feature_names: List[str]
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        feature_names = [str(x) for x in cfg.get("feature_names", [])]
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # FIX 5: raise error instead of silently falling back on mismatch
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_config.json has {len(feature_names)} feature names "
            f"but X_features.npy has {X.shape[1]} columns. "
            f"Re-run extract_features.py to regenerate feature_config.json."
        )

    # stratified split
    print("\nSplitting data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_seed,
        )
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

    # ── model 1: LR L2 ────────────────────────────────────────────────
    lr_l2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,              # FIX 6: bumped from 3000
            n_jobs=-1,
            multi_class="auto",
        )),
    ])
    # FIX 4: capture returned model consistently
    metrics_lr_l2, _, fitted_lr_l2 = evaluate_and_save(
        model_name="lr_l2",
        model=lr_l2,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    # ── model 2: LR L1 ────────────────────────────────────────────────
    lr_l1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            n_jobs=-1,
            tol=1e-3,
            C=1.0,
            multi_class="ovr",
        )),
    ])
    # FIX 1: capture returned model — no second training needed
    metrics_lr_l1, _, fitted_lr_l1 = evaluate_and_save(
        model_name="lr_l1",
        model=lr_l1,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    # extract sparsity from already-fitted model — no retraining
    coef     = fitted_lr_l1.named_steps["clf"].coef_
    sparsity = l1_sparsity_report(coef=coef, feature_names=feature_names)
    _save_json(os.path.join(args.results_dir, "lr_l1_sparsity.json"), sparsity)
    # FIX 7: print selected features on one line instead of one per line
    print(f"\n  L1 selected {sparsity['num_selected_features']} / "
          f"{sparsity['feature_dim']} features: "
          f"{', '.join(sparsity['selected_feature_names'])}")

    # ── model 3: linear SVM ───────────────────────────────────────────
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            random_state=args.random_seed,
            max_iter=20000,
        )),
    ])
    # FIX 5: capture returned model consistently
    metrics_svm, _, fitted_svm = evaluate_and_save(
        model_name="svm_linear",
        model=svm,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        label_names=label_names,
        results_dir=args.results_dir,
    )

    # FIX 6: print final summary table instead of just "done"
    all_metrics = [metrics_lr_l2, metrics_lr_l1, metrics_svm]
    print("\n" + "=" * 57)
    print(f"{'Model':<15} {'Accuracy':>10} {'Macro-F1':>10} {'Time':>12}")
    print("=" * 57)
    for m in all_metrics:
        print(
            f"{m['model']:<15} "
            f"{m['accuracy']:>10.4f} "
            f"{m['macro_f1']:>10.4f} "
            f"{m['train_time_sec']:>11.0f}s"
        )
    print("=" * 57)
    print(f"\nAll done. Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()