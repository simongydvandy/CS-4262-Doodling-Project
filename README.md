# CS-4262-Doodling-Project
We will build a “Pictionary” classifier that guesses the object category of a sketch. The main learning goal is to compare classic course models (logistic regression, SVM, random forests) on engineered stroke features versus a CNN on rasterized sketches and quantify how much the CNN helps. Dataset: https://quickdraw.withgoogle.com/data

## Getting Started (Environment Setup)

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies reliably.

1. **Install `uv`** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install all dependencies**:
   ```bash
   uv sync
   ```
   *This single command reads `uv.lock`, creates a `.venv` virtual environment, and identically strictly installs Jupyter, numpy, scikit-learn, PyTorch, etc.*

   `.venv` is intended to be managed by `uv`. Do not create or update it manually with `python -m venv` / `pip install` if you want reproducible results.

3. **Run your code**:
   Prefix commands with `uv run` to automatically run code inside the isolated environment.
   ```bash
   uv run jupyter notebook
   ```

---

## How to download processed data

### Run the download script
```bash
# For CNN models (X_cnn.npy, y_cnn.npy — ~521 MB)
uv run python download_data.py --role cnn

# For baseline models (strokes.ndjson — ~327 MB)
uv run python download_data.py --role baseline
# If Google Drive download fails with SSL/certificate verification:
# uv run python download_data.py --role baseline --no-ssl-verify

# For everything (~848 MB)
uv run python download_data.py --role all
```

Files will be saved to `data/processed/`.

### 3. Re-downloading
If the dataset has been updated, simply re-run the script. Already existing files will be skipped automatically. To force a fresh download, delete the files in `data/processed/` first.

> **Note:** The dataset and outputs (large `*.npy`/`*.json` files) are generated locally and should not be committed to the repo.

## Stroke feature engineering + baseline training

The raw download script fetches `data/processed/strokes.ndjson` (and optionally CNN arrays), but it does not build the engineered “stroke feature matrix”.
Use the provided scripts below:

### Step 1 — Build `X_features.npy` (and labels)
```bash
# Full run
uv run python extract_features.py

# Quick smoke test
uv run python extract_features.py --limit 10000
```

This reads `data/processed/strokes.ndjson` and writes:
- `data/processed/X_features.npy` (shape: `N x 18`)
- `data/processed/y_features.npy` (shape: `N`)
- `data/processed/label_names.json` (class name per label id)
- `data/processed/feature_config.json` (feature/dimension config)

### Step 2 — Train baseline classifiers
```bash
# Fast test on the first K categories (labels 0..K-1)
uv run python train_baseline.py --categories 10

# Full run (690k samples)
uv run python train_baseline.py
```

This trains:
- Logistic Regression (L2)
- Logistic Regression (L1, used for sparsity/feature selection reporting)
- Linear SVM

It saves metrics and confusion matrix plots to `results/`.

## CNN training on rasterized sketches

The repo also includes a PyTorch CNN baseline for the rasterized QuickDraw images (`X_cnn.npy`, `y_cnn.npy`).
This is meant to support the project comparison between classical engineered-feature models and a learned image model.

### Step 1 - Install PyTorch

If you are running locally:
```bash
uv sync
```

If you are running in Google Colab, PyTorch is usually preinstalled already.

### Step 2 - Train the CNN
```bash
# Quick smoke test on 10 categories
uv run python train_cnn.py --categories 10 --limit 10000 --epochs 5

# Full run
uv run python train_cnn.py --epochs 15 --batch-size 256
```

By default, the script:
- loads `data/processed/X_cnn.npy` and `data/processed/y_cnn.npy`
- normalizes images to `[0, 1]`
- creates stratified train/validation/test splits
- trains a LeNet-style CNN
- selects the best checkpoint by validation macro-F1

### Step 3 - Review outputs

The CNN script writes the following files to `results/`:
- `cnn_best.pt` - best model checkpoint
- `cnn_metrics.json` - summary metrics
- `cnn_classification_report.json` - per-class precision/recall/F1
- `cnn_confusion_matrix.npy` and `cnn_confusion_matrix.png`
- `cnn_history.json` - epoch-by-epoch metrics
- `cnn_training_curves.png`

Useful knobs for experiments:
```bash
uv run python train_cnn.py \
  --batch-size 512 \
  --learning-rate 5e-4 \
  --dropout 0.4 \
  --hidden-dim 256 \
  --conv-channels 32 64 \
  --class-weighting
```

This makes it easy to run one stable baseline in the repo while trying variants in Colab.
