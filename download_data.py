# scripts/download_data.py
import gdown
import os
import argparse

# ── Folder ID ────────────────────────────────────────────────
PROCESSED_FOLDER_ID = "1OhGaaKRE2dMPaYNvK7cvht0VNE9rq6Kf"

# ── Role definitions ─────────────────────────────────────────
ROLES = {
    "cnn":      ["X_cnn.npy", "y_cnn.npy"],
    "baseline": ["strokes.ndjson"],
    "all":      ["X_cnn.npy", "y_cnn.npy", "strokes.ndjson"],
}

# ── File IDs (from Google Drive) ─────────────────────────────
FILE_IDS = {
    "strokes.ndjson": "1FxKyfwJRyQzRNOeRUqbtWhBX2bwFAC3S",
    "X_cnn.npy":      "1HfFQNydRtS7-AURmX_L01Y_lfsVmHSxm",
    "y_cnn.npy":      "1IEqzY6WYceUNzIcrfYT058Fs1ibNPx-h",
}

def download(role="all"):
    if role not in ROLES:
        print(f"Unknown role. Choose from: {list(ROLES.keys())}")
        return

    os.makedirs("data/processed", exist_ok=True)
    files_needed = ROLES[role]

    print(f"\nDownloading files for role: '{role}'")
    print(f"Files: {files_needed}\n")

    for filename in files_needed:
        out_path = f"data/processed/{filename}"

        if os.path.exists(out_path):
            print(f"  ✓ Already exists, skipping: {filename}")
            continue

        print(f"  ↓ Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_IDS[filename]}",
            output=out_path,
            quiet=False
        )

    print("\nDone! Files saved to data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=["cnn", "baseline", "all"],
        default="all",
        help=(
            "Which files to download:\n"
            "  cnn      → X_cnn.npy, y_cnn.npy   (~521 MB)\n"
            "  baseline → strokes.ndjson          (~327 MB)\n"
            "  all      → everything              (~848 MB)\n"
        )
    )
    args = parser.parse_args()
    download(args.role)