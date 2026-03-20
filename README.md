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
   *This single command reads `uv.lock`, creates a `.venv` virtual environment, and identically strictly installs Jupyter, numpy, scikit-learn, etc.*

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
