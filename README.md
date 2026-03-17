# CS-4262-Doodling-Project
We will build a “Pictionary” classifier that guesses the object category of a sketch. The main learning goal is to compare classic course models (logistic regression, SVM, random forests) on engineered stroke features versus a CNN on rasterized sketches and quantify how much the CNN helps. Dataset: https://quickdraw.withgoogle.com/data

## How to download processed data

### 1. Install dependency
```bash
pip install gdown
```

### 2. Run the download script
```bash
# For CNN models (X_cnn.npy, y_cnn.npy — ~521 MB)
python scripts/download_data.py --role cnn

# For baseline models (strokes.ndjson — ~327 MB)
python scripts/download_data.py --role baseline

# For everything (~848 MB)
python scripts/download_data.py --role all
```

Files will be saved to `data/processed/`.

### 3. Re-downloading
If the dataset has been updated, simply re-run the script. Already existing files will be skipped automatically. To force a fresh download, delete the files in `data/processed/` first.

> **Note:** `data/` is in `.gitignore` — do not commit dataset files to the repo.
