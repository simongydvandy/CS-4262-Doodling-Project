# CS-4262-Doodling-Project

**Classname:** CS4262 - Foundations of Machine Learning  
**Collaborators:** Mohamed Bakry, Simon Gou, Xinran Shi, John Abad, Rojin Sharma

## Introduction

This project builds a sketch-classification system inspired by Pictionary and Google Quick, Draw!. The goal is to predict the object category of a drawing and compare two modeling approaches:

- classical machine learning models trained on engineered stroke features
- convolutional neural networks trained on rasterized sketch images

The main question behind the project is how much performance we gain by moving from hand-crafted features to learned visual representations.

Dataset source: [Google Quick, Draw!](https://quickdraw.withgoogle.com/data)

## Project Structure

- `scripts/` contains the main download, preprocessing, and training scripts
- `docs/` contains runbooks, milestone notes, and experiment documentation
- `notebooks/` contains Colab and exploratory notebooks
- `data/` stores downloaded and processed data locally and is not committed
- `results/` stores checkpoints, metrics, and plots locally and is not committed

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for reproducible environment management.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

This creates a `uv`-managed `.venv` and installs the locked dependency set from `uv.lock`.

### Run commands

Use `uv run` so commands execute inside the project environment:

```bash
uv run python --version
```

## How To Run

### 1. Download the data

Use the download script to fetch the processed files needed for either the CNN pipeline, the classical baseline pipeline, or both.

```bash
# CNN arrays only
uv run python scripts/download_quickdraw_data.py --role cnn

# Stroke data only
uv run python scripts/download_quickdraw_data.py --role baseline

# Everything
uv run python scripts/download_quickdraw_data.py --role all
```

If Google Drive download fails because of SSL verification on your network:

```bash
uv run python scripts/download_quickdraw_data.py --role baseline --no-ssl-verify
```

Downloaded files are stored in `data/processed/`.

### 2. Build engineered stroke features

The classical models do not train directly on `strokes.ndjson`; they use a feature matrix built from the stroke sequences.

```bash
# Full feature extraction
uv run python scripts/build_stroke_features.py

# Small smoke test
uv run python scripts/build_stroke_features.py --limit 10000
```

This creates:

- `data/processed/X_features.npy`
- `data/processed/y_features.npy`
- `data/processed/label_names.json`
- `data/processed/feature_config.json`

The current feature extractor produces a 24-dimensional feature vector per sketch.

### 3. Train the classical baselines

```bash
# Quick test on first 10 classes
uv run python scripts/train_classical_models.py --categories 10

# Full run
uv run python scripts/train_classical_models.py
```

The classical pipeline trains:

- Logistic Regression (L2)
- Logistic Regression (L1)
- Linear SVM

Artifacts such as metrics, confusion matrices, and saved pipelines are written locally to `results/` and `models/`.

#### Distributed C-tuning (team parallel search)

`C` controls inverse regularization strength (`C=1.0` default). Rather than running an expensive grid search on one machine, each teammate runs one value and the team compares the resulting `*_metrics.json` files. Each run takes roughly 1–2 hours on a laptop.

**Prerequisites:** everyone needs the same `X_features.npy` / `y_features.npy` — either run `build_stroke_features.py` locally or share the files. The `--results-dir` flag keeps each run's outputs isolated so nothing is overwritten.

```bash
# Teammate 1
uv run python scripts/train_classical_models.py --C 0.01 --results-dir results/C_0.01

# Teammate 2
uv run python scripts/train_classical_models.py --C 0.1  --results-dir results/C_0.1

# Teammate 3  (current default — reuse existing results if already run)
uv run python scripts/train_classical_models.py --C 1.0  --results-dir results/C_1.0

# Teammate 4
uv run python scripts/train_classical_models.py --C 10   --results-dir results/C_10

# Teammate 5
uv run python scripts/train_classical_models.py --C 100  --results-dir results/C_100
```

Share the three small `*_metrics.json` files from each run, pick the C with the highest `macro_f1`, then do one final run with that C:

```bash
uv run python scripts/train_classical_models.py --C <best_C>
```

### 4. Train the CNN pipeline

The CNN pipeline uses rasterized QuickDraw arrays from `X_cnn.npy` and `y_cnn.npy`.

```bash
# Small smoke test
uv run python scripts/train_sketch_cnn.py --categories 10 --limit 10000 --epochs 5

# Standard full-data run
uv run python scripts/train_sketch_cnn.py --epochs 15 --batch-size 256
```

The CNN trainer supports multiple architectures:

- `lenet`
- `deep`
- `resnet`

It also supports:

- class weighting
- data augmentation
- learning-rate schedulers
- top-1, top-3, and top-5 evaluation metrics

Example of a stronger ResNet run:

```bash
uv run python scripts/train_sketch_cnn.py \
  --model-type resnet \
  --augment \
  --scheduler plateau \
  --epochs 30 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --hidden-dim 512 \
  --conv-channels 64 128 256 \
  --dropout 0.35
```

## Outputs

### Classical-model outputs

The classical pipeline writes metrics and confusion matrices to `results/`, and saved model pipelines to `models/`.

### CNN outputs

The CNN pipeline writes these artifacts to `results/`:

- `cnn_best.pt`
- `cnn_metrics.json`
- `cnn_classification_report.json`
- `cnn_per_class_metrics.csv`
- `cnn_presentation_summary.json`
- `cnn_history.json`
- `cnn_confusion_matrix.npy`
- `cnn_confusion_matrix.png`
- `cnn_training_curves.png`

These outputs include:

- top-1, top-3, and top-5 accuracy
- macro-F1
- per-class precision, recall, and F1
- most common confusion pairs
- weakest classes by F1
- full experiment configuration

## Recommended Workflow

For day-to-day work, a good sequence is:

1. Download the needed data.
2. Run a small smoke test first.
3. Train the baseline classical models.
4. Train the CNN or ResNet models.
5. Compare metrics, confusion matrices, and per-class behavior.
6. Save the best run artifacts for the report and presentation.

## Colab / Experiment References

- [`docs/CNN_COLAB_RUNBOOK.md`](docs/CNN_COLAB_RUNBOOK.md)
- [`notebooks/cnn_colab_runner.ipynb`](notebooks/cnn_colab_runner.ipynb)
- [`notebooks/quickdraw_data_download.ipynb`](notebooks/quickdraw_data_download.ipynb)

## Notes

- Large generated files in `data/` and `results/` should stay out of Git.
- The raster CNN input file may be stored as flattened `(N, 784)` vectors; the training script reshapes this automatically.
- If Colab has trouble with workers, rerun with `--num-workers 0`.
- For reproducibility, prefer `uv sync` and `uv run ...` over a manually managed virtual environment.
