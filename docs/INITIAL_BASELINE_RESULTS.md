# Classical Baseline Results

This file records the smoke test sanity check and the best full-run results for the classical baseline models (24-feature vector, C=10, class_weight='balanced').

---

## Smoke Test (10 categories)

A quick sanity check on the first 10 categories before committing to the full 690K-sample run.

```bash
uv run python scripts/build_stroke_features.py --limit 10000
uv run python scripts/train_classical_models.py --C 10 --categories 10
```

Test set: 3,603 samples across 10 classes.

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| lr_l2 | **0.7485** | **0.7458** |
| lr_l1 | 0.7386 | 0.7354 |
| svm_linear | 0.7277 | 0.7195 |

L1 sparsity: 24 / 24 features selected (all features carry signal).

---

## Full Run (345 categories)

Two C values were tested across teammates. Test set: 126,590 samples across 345 classes.

```bash
uv run python scripts/build_stroke_features.py
uv run python scripts/train_classical_models.py --C 10  --results-dir results/C_10
uv run python scripts/train_classical_models.py --C 100 --results-dir results/C_100
```

| Model | C=10 Acc | C=10 F1 | C=100 Acc | C=100 F1 |
|-------|----------|---------|-----------|----------|
| lr_l2 | **0.2477** | **0.2248** | 0.2477 | 0.2249 |
| lr_l1 | 0.2084 | 0.1716 | 0.2085 | 0.1718 |
| svm_linear | 0.1872 | 0.1260 | 0.1872 | 0.1260 |

**Best model: LR L2** — 0.2477 accuracy, 0.2248 macro-F1 at both C=10 and C=100.

C=10 and C=100 produce virtually identical results, meaning LR L2 has plateaued and further relaxing regularization provides no benefit. C=10 is the recommended value.

Random baseline on 345 classes = 1/345 = 0.29%. LR L2 is ~85× better than random.

The ~25% ceiling is expected for linear models on 24 hand-engineered stroke features across 345 categories. Many QuickDraw categories share similar global stroke statistics and cannot be separated by a linear boundary in this feature space. Higher accuracy requires a non-linear model such as the CNN.

---

## Where the confusion matrices are saved

Running `scripts/train_classical_models.py` writes confusion matrices to the local `results/` directory (PNG + NPY). Those artifacts are intentionally not committed to the repo.
