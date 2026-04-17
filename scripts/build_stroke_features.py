import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────

FEATURE_DIM = 24

FEATURE_NAMES = [
    # Family 1 — Counting (4)
    "num_strokes",
    "num_points",
    "avg_points_per_stroke",
    "max_points_per_stroke",        # renamed from max_stroke_len (was misleading — counts points, not spatial length)
    # Family 2 — Geometry (6)
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "log_bbox_area",                # was bbox_area; log1p applied to compress 6-order-of-magnitude range
    "centroid_x_norm",              # was centroid_x (absolute canvas coords); now normalized to [0,1] within bbox
    "centroid_y_norm",              # was centroid_y (absolute canvas coords); now normalized to [0,1] within bbox
    # Family 3 — Ink (3)
    "ink_length_total",
    "ink_length_mean_segment",
    "ink_length_norm_diag",
    # Family 4 — Direction (8 bins + entropy; dominant_direction removed — circular index was mis-encoded as continuous)
    "dir_hist_bin_0",
    "dir_hist_bin_1",
    "dir_hist_bin_2",
    "dir_hist_bin_3",
    "dir_hist_bin_4",
    "dir_hist_bin_5",
    "dir_hist_bin_6",
    "dir_hist_bin_7",
    "direction_entropy",
    # Family 5 — Curvature (2)
    "corners_per_stroke",           # was corners_count (raw); now corners_count / num_strokes — more meaningful
    "corners_density",
    # NOTE: 'recognized' removed — default pipeline filters to recognized=True only, making this constant (always 1.0)
    # NOTE: 'dominant_direction' removed — argmax index 0–7 is circular (bin 7 ≈ bin 0) but was treated as continuous
]

assert len(FEATURE_NAMES) == FEATURE_DIM


# ── helpers ────────────────────────────────────────────────────────────────────

def _direction_histogram(angles_rad: List[float], bins: int = 8) -> np.ndarray:
    """
    Return a normalized direction histogram over [0, 2pi).

    Maps atan2 output (which lives in [-pi, pi]) into [0, 2pi)
    before binning so all angles land in a valid bucket.
    """
    if not angles_rad:
        return np.zeros(bins, dtype=np.float32)
    two_pi = 2.0 * math.pi
    angles = (np.asarray(angles_rad, dtype=np.float64) + two_pi) % two_pi
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, two_pi))
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    return hist / s if s > 0 else hist


def _count_corners(
    xs: List[float],
    ys: List[float],
    threshold_rad: float,
) -> int:
    """
    Count corner-like points using turning angle threshold.

    For each consecutive triple of points (p_prev, p_curr, p_next),
    computes the angle between the incoming and outgoing vectors using
    the dot product. Counts as a corner if that angle >= threshold.
    Uses dot product instead of atan2 differences to avoid wraparound issues.
    """
    n = min(len(xs), len(ys))
    if n < 3:
        return 0
    corners = 0
    for i in range(1, n - 1):
        v1x = xs[i] - xs[i - 1]
        v1y = ys[i] - ys[i - 1]
        v2x = xs[i + 1] - xs[i]
        v2y = ys[i + 1] - ys[i]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0.0 or n2 == 0.0:
            continue
        cosang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        if math.acos(cosang) >= threshold_rad:
            corners += 1
    return corners


# ── main feature extractor ─────────────────────────────────────────────────────

def compute_stroke_features(
    drawing: List,
    direction_bins: int = 8,
    corner_angle_deg: float = 30.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a 24-dim feature vector from a QuickDraw simplified stroke drawing.

    The drawing format expected is the simplified ndjson format:
        drawing = [
            [[x0, x1, x2, ...], [y0, y1, y2, ...]],   # stroke 1
            [[x0, x1, x2, ...], [y0, y1, y2, ...]],   # stroke 2
            ...
        ]

    Features (24):
        Family 1 — Counting (4):
            0  num_strokes
            1  num_points
            2  avg_points_per_stroke
            3  max_points_per_stroke       (renamed from max_stroke_len)

        Family 2 — Geometry (6):
            4  bbox_width
            5  bbox_height
            6  bbox_aspect_ratio
            7  log_bbox_area               (log1p of bbox_area; was bbox_area — huge range compressed)
            8  centroid_x_norm             (was centroid_x; now (centroid_x - minx) / bbox_width ∈ [0,1])
            9  centroid_y_norm             (was centroid_y; now (centroid_y - miny) / bbox_height ∈ [0,1])

        Family 3 — Ink (3):
            10  ink_length_total
            11  ink_length_mean_segment
            12  ink_length_norm_diag

        Family 4 — Direction (9):
            13-20  dir_hist_bin_0..7
            21     direction_entropy
            (dominant_direction removed — circular argmax index was mis-encoded as a continuous feature)

        Family 5 — Curvature (2):
            22  corners_per_stroke         (was corners_count; now corners_count / num_strokes)
            23  corners_density
    """
    corner_threshold = corner_angle_deg * math.pi / 180.0

    # ── collect raw data across all strokes ────────────────────────────
    all_x: List[float] = []
    all_y: List[float] = []
    angles: List[float] = []
    segment_lengths: List[float] = []
    stroke_point_counts: List[int] = []
    total_corners = 0
    total_points = 0

    for stroke in (drawing or []):
        if not stroke:
            continue

        # simplified ndjson format is always [[x0,x1,...], [y0,y1,...]]
        xs_seq = [float(v) for v in stroke[0]]
        ys_seq = [float(v) for v in stroke[1]]
        n = min(len(xs_seq), len(ys_seq))
        if n <= 0:
            continue

        total_points += n
        stroke_point_counts.append(n)
        all_x.extend(xs_seq[:n])
        all_y.extend(ys_seq[:n])

        # corners per stroke
        total_corners += _count_corners(xs_seq[:n], ys_seq[:n], corner_threshold)

        # segment directions and lengths
        if n < 2:
            continue
        for i in range(n - 1):
            dx = xs_seq[i + 1] - xs_seq[i]
            dy = ys_seq[i + 1] - ys_seq[i]
            length = math.hypot(dx, dy)
            if length == 0.0:
                continue
            segment_lengths.append(length)
            angles.append(math.atan2(dy, dx))

    # ── guard: return zeros for empty drawings ─────────────────────────
    if not all_x:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # ── family 1: counting ─────────────────────────────────────────────
    num_strokes              = len(stroke_point_counts)
    num_points               = total_points
    avg_points_per_stroke    = float(num_points) / (num_strokes + eps)
    max_points_per_stroke    = float(max(stroke_point_counts))  # renamed from max_stroke_len

    # ── family 2: geometry ─────────────────────────────────────────────
    minx, maxx   = min(all_x), max(all_x)
    miny, maxy   = min(all_y), max(all_y)
    bbox_width   = float(maxx - minx)
    bbox_height  = float(maxy - miny)
    bbox_aspect  = bbox_width / (bbox_height + eps)
    # log1p compresses the 6-order-of-magnitude range of bbox_area so outliers don't dominate after StandardScaler
    log_bbox_area = math.log1p(bbox_width * bbox_height)
    raw_cx        = float(sum(all_x) / len(all_x))
    raw_cy        = float(sum(all_y) / len(all_y))
    # Normalize centroid to [0,1] within the bounding box so position encodes shape, not canvas location
    centroid_x_norm = (raw_cx - minx) / (bbox_width  + eps)
    centroid_y_norm = (raw_cy - miny) / (bbox_height + eps)

    # ── family 3: ink ──────────────────────────────────────────────────
    ink_total        = float(sum(segment_lengths)) if segment_lengths else 0.0
    num_segments     = len(segment_lengths)
    ink_mean         = ink_total / (num_segments + eps)
    bbox_diag        = math.hypot(bbox_width, bbox_height)
    ink_norm_diag    = ink_total / (bbox_diag + eps)

    # ── family 4: direction ────────────────────────────────────────────
    hist    = _direction_histogram(angles, bins=direction_bins)
    # dominant_direction (argmax index) removed: bins 0–7 represent circular angles, so the
    # index is NOT a continuous value (bin 7 ≈ bin 0). The 8 histogram bins already carry
    # this information without the encoding bug.
    entropy = float(-np.sum(hist * np.log(hist + eps)))

    # ── family 5: curvature ────────────────────────────────────────────
    # corners_per_stroke normalizes by num_strokes so complex drawings with many strokes
    # aren't unfairly penalized vs. simple single-stroke drawings
    corners_per_stroke = float(total_corners) / float(num_strokes + eps)
    corners_density    = float(total_corners) / float(max(total_points - 2, 1))

    # ── assemble feature vector ────────────────────────────────────────
    feats = np.array([
        # family 1 — counting
        float(num_strokes),
        float(num_points),
        float(avg_points_per_stroke),
        float(max_points_per_stroke),
        # family 2 — geometry
        float(bbox_width),
        float(bbox_height),
        float(bbox_aspect),
        float(log_bbox_area),
        float(centroid_x_norm),
        float(centroid_y_norm),
        # family 3 — ink
        float(ink_total),
        float(ink_mean),
        float(ink_norm_diag),
        # family 4 — direction histogram + entropy (dominant_direction removed)
        *hist.tolist(),
        float(entropy),
        # family 5 — curvature
        float(corners_per_stroke),
        float(corners_density),
        # 'recognized' removed — always 1.0 when using default filter, so zero-variance after scaling
    ], dtype=np.float32)

    assert feats.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM} features, got {feats.shape}"
    return feats


# ── loader ─────────────────────────────────────────────────────────────────────

def load_features(
    input_path: str,
    include_unrecognized: bool = True,
    direction_bins: int = 8,
    corner_angle_deg: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Read strokes.ndjson and extract features for every record.

    Uses pre-allocated arrays instead of list accumulation to avoid
    the RAM spike that comes from stacking 690K numpy arrays at the end.
    Prints progress every 10K records.

    Returns:
        X            : np.ndarray of shape (N, 26), dtype float32
        y            : np.ndarray of shape (N,),    dtype int64
        label_names  : list of category name strings indexed by label int
        config       : dict with metadata about the extraction run
    """
    # ── count total lines for pre-allocation ───────────────────────────
    print("Counting records...")
    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)
    print(f"  Found {total_lines:,} records\n")

    # ── pre-allocate ───────────────────────────────────────────────────
    X = np.zeros((total_lines, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(total_lines, dtype=np.int64)

    label_to_name: Dict[int, str] = {}
    processed = 0

    # ── main extraction loop ───────────────────────────────────────────
    with open(input_path, "r") as f:
        for line in f:
            rec = json.loads(line)

            recognized = bool(rec.get("recognized", True))
            if not include_unrecognized and not recognized:
                continue

            label = int(rec["label"])
            label_to_name[label] = str(rec["category"])

            X[processed] = compute_stroke_features(
                rec.get("drawing", []),
                direction_bins=direction_bins,
                corner_angle_deg=corner_angle_deg,
            )
            y[processed] = label
            processed += 1

            if processed % 10000 == 0:
                pct = processed / total_lines * 100
                print(f"  {processed:>7,} / {total_lines:,}  ({pct:.1f}%)")

    # ── trim in case rows were skipped by unrecognized filter ──────────
    X = X[:processed]
    y = y[:processed]

    # ── build label name list indexed by label int ─────────────────────
    if label_to_name:
        max_label  = max(label_to_name.keys())
        label_names = [label_to_name.get(i, f"label_{i}") for i in range(max_label + 1)]
    else:
        label_names = []

    config = {
        "input_path":               input_path,
        "include_unrecognized":     include_unrecognized,
        "direction_bins":           direction_bins,
        "corner_angle_deg":         corner_angle_deg,
        "feature_dim":              FEATURE_DIM,
        "feature_names":            FEATURE_NAMES,
        "num_samples":              int(X.shape[0]),
        "label_cardinality":        int(len(label_to_name)),
    }

    print(f"\nDone. Extracted {processed:,} records")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    return X, y, label_names, config


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 26 stroke features from strokes.ndjson."
    )
    parser.add_argument(
        "--input",
        default="data/processed/strokes.ndjson",
        help="Path to strokes.ndjson",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write X_features.npy, y_features.npy, label_names.json, feature_config.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records to process (useful for quick testing).",
    )
    parser.add_argument(
        "--include-unrecognized",
        action="store_true",
        help="Include records where recognized=False (default: excluded)",
    )
    parser.add_argument(
        "--direction-bins",
        type=int,
        default=8,
        help="Number of direction histogram bins (must be 8 for 26-dim output).",
    )
    parser.add_argument(
        "--corner-angle-deg",
        type=float,
        default=30.0,
        help="Turning angle threshold in degrees for corner counting.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, label_names, config = load_features(
        args.input,
        include_unrecognized=args.include_unrecognized,
        direction_bins=args.direction_bins,
        corner_angle_deg=args.corner_angle_deg,
    )

    # ── apply limit after extraction (for testing) ─────────────────────
    if args.limit is not None:
        X = X[:args.limit]
        y = y[:args.limit]
        print(f"Limit applied: using first {args.limit:,} records")

    # ── save outputs ───────────────────────────────────────────────────
    X_path      = os.path.join(args.output_dir, "X_features.npy")
    y_path      = os.path.join(args.output_dir, "y_features.npy")
    label_path  = os.path.join(args.output_dir, "label_names.json")
    config_path = os.path.join(args.output_dir, "feature_config.json")

    np.save(X_path, X)
    np.save(y_path, y)

    with open(label_path, "w") as f:
        json.dump(label_names, f, indent=2)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved:")
    print(f"  {X_path}")
    print(f"  {y_path}")
    print(f"  {label_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()