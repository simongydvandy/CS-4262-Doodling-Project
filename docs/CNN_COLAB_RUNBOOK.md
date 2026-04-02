# CNN Colab Runbook

This runbook is for running the QuickDraw CNN pipeline in Google Colab and collecting results for the final project comparison.

## Goal

Train the CNN on the rasterized QuickDraw data, compare it against the existing classical baselines, and save the best run artifacts for the report.

## Recommended workflow

1. Open the Colab notebook `notebooks/cnn_colab_runner.ipynb`.
2. Run the smoke test first to verify the environment and data pipeline.
3. Run one baseline full-data CNN experiment.
4. Run 2-4 controlled variants by changing one or two hyperparameters at a time.
5. Save the best metrics, confusion matrix, and training curves.

## Smoke test

Use this first:

```bash
python scripts/train_cnn.py --categories 10 --limit 10000 --epochs 5 --batch-size 256 --num-workers 0
```

Expected outcome:
- script completes without shape/device errors
- validation metrics improve over epochs
- files appear in `results/`

## First full run

```bash
python scripts/train_cnn.py --epochs 15 --batch-size 256 --num-workers 2
```

If classes seem imbalanced in the results, also try:

```bash
python scripts/train_cnn.py --epochs 15 --batch-size 256 --class-weighting --num-workers 2
```

## Suggested experiment grid

Try a few focused variants instead of too many random combinations.

### Baseline

```bash
python scripts/train_cnn.py --epochs 15 --batch-size 256 --learning-rate 1e-3 --dropout 0.3 --hidden-dim 128 --conv-channels 32 64
```

### Variant A: lower learning rate

```bash
python scripts/train_cnn.py --epochs 20 --batch-size 256 --learning-rate 5e-4 --dropout 0.3 --hidden-dim 128 --conv-channels 32 64
```

### Variant B: stronger regularization

```bash
python scripts/train_cnn.py --epochs 20 --batch-size 256 --learning-rate 1e-3 --dropout 0.4 --hidden-dim 128 --conv-channels 32 64
```

### Variant C: wider model

```bash
python scripts/train_cnn.py --epochs 15 --batch-size 256 --learning-rate 1e-3 --dropout 0.3 --hidden-dim 256 --conv-channels 64 128
```

### Variant D: class weighting

```bash
python scripts/train_cnn.py --epochs 15 --batch-size 256 --learning-rate 1e-3 --dropout 0.3 --hidden-dim 128 --conv-channels 32 64 --class-weighting
```

## What to save after each run

Copy or download these from `results/`:

- `cnn_metrics.json`
- `cnn_classification_report.json`
- `cnn_confusion_matrix.png`
- `cnn_training_curves.png`
- `cnn_best.pt`

Rename them per experiment if needed so runs do not overwrite each other.

## What to record in the report

For each meaningful run, record:

- command used
- runtime
- best validation macro-F1
- test accuracy
- test macro-F1
- notable confused classes

## Final comparison table

Include at least:

- Logistic Regression (L2)
- Logistic Regression (L1)
- Linear SVM
- Best CNN run

Metrics:

- accuracy
- macro-F1
- training time
- notes on strengths/weaknesses

## Notes

- The real `X_cnn.npy` file is stored as flattened `28x28` images with shape `(N, 784)`. The training script already reshapes this automatically.
- On Colab, if DataLoader workers cause issues, rerun with `--num-workers 0`.
- If a run is interrupted, keep the generated `cnn_best.pt` checkpoint and saved metrics.
