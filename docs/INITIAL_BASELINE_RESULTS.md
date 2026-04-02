# Initial Baseline Results (sanity check)

This file records a small smoke test run so collaborators can quickly see the baseline performance before the full 690k-sample training.

## How the run was produced

Feature engineering:
```bash
python scripts/extract_features.py --limit 10000
```

Baseline training:
```bash
python scripts/train_baseline.py --categories 10
```

Notes / defaults used by the scripts:
- Only `recognized=true` records are used in `scripts/extract_features.py` (unless `--include-unrecognized` is passed).
- `scripts/extract_features.py` outputs an 18-dimensional feature vector per sample (10 numeric/geometric features + an 8-bin direction histogram).
- `scripts/train_baseline.py` uses:
  - `--test-size 0.2` (default)
  - `--random-seed 42` (default)
  - `stratify=y` when possible

## Metrics (10 categories)

Test set size (after filtering to labels `< 10`): `58` samples across `10` classes.

- `lr_l2`
  - accuracy: `0.603448275862069`
  - macro_f1: `0.5927738927738928`
- `lr_l1` (L1 sparsity / feature selection reporting)
  - accuracy: `0.6206896551724138`
  - macro_f1: `0.6056654456654458`
- `svm_linear` (linear SVM)
  - accuracy: `0.603448275862069`
  - macro_f1: `0.5829109779109778`

## L1 sparsity report (from `lr_l1`)

- Non-zero threshold (`eps`): `1e-06`
- Selected features: `18 / 18` (all features had non-zero coefficients above the threshold)
- Selected feature names:
  - `ink_length_total`
  - `ink_length_mean_segment`
  - `num_strokes`
  - `num_points`
  - `bbox_width`
  - `bbox_height`
  - `bbox_aspect_ratio`
  - `ink_length_norm_diag`
  - `corners_count`
  - `corners_density`
  - `dir_hist_bin_0`
  - `dir_hist_bin_1`
  - `dir_hist_bin_2`
  - `dir_hist_bin_3`
  - `dir_hist_bin_4`
  - `dir_hist_bin_5`
  - `dir_hist_bin_6`
  - `dir_hist_bin_7`

## Where the confusion matrices are saved

When you run `scripts/train_baseline.py`, it writes confusion matrices to the local `results/` directory (PNG + NPY). Those `results/` artifacts are intentionally not committed to the repo.
