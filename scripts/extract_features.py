import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _direction_histogram(angles_rad: List[float], bins: int = 8) -> np.ndarray:
    """Return a normalized direction histogram over [0, 2pi)."""
    if not angles_rad:
        return np.zeros((bins,), dtype=np.float32)

    two_pi = 2.0 * math.pi
    # Map angles to [0, 2pi)
    angles = (np.asarray(angles_rad, dtype=np.float64) + two_pi) % two_pi
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, two_pi))
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    return hist / s if s > 0 else hist


def _count_corners_from_sequences(
    xs: List[float],
    ys: List[float],
    corner_angle_rad: float,
) -> int:
    """Count corner-like points based on turning angle threshold."""
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

        # Angle between v1 and v2
        dot = v1x * v2x + v1y * v2y
        cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
        ang = math.acos(cosang)

        if ang >= corner_angle_rad:
            corners += 1
    return corners


def compute_stroke_features(
    drawing: List,
    *,
    direction_bins: int = 8,
    corner_angle_deg: float = 30.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute an 18-dim feature vector from a QuickDraw simplified stroke drawing.

    Features (18):
      0  ink_length_total
      1  ink_length_mean_segment
      2  num_strokes
      3  num_points
      4  bbox_width
      5  bbox_height
      6  bbox_aspect_ratio
      7  ink_length_norm_diag
      8  corners_count
      9  corners_density
      10..17 direction_histogram (8 bins, normalized)
    """
    corner_angle_rad = float(corner_angle_deg) * math.pi / 180.0

    num_strokes = len(drawing) if drawing is not None else 0

    xs: List[float] = []
    ys: List[float] = []
    angles: List[float] = []
    segment_lengths: List[float] = []
    total_corners = 0
    total_points = 0

    for stroke in (drawing or []):
        if not stroke:
            continue

        # QuickDraw simplified format is typically: stroke = [xs, ys],
        # where xs and ys are sequences of coordinates.
        xs_seq: List[float]
        ys_seq: List[float]
        if (
            isinstance(stroke, list)
            and len(stroke) == 2
            and isinstance(stroke[0], list)
            and isinstance(stroke[1], list)
            and all(isinstance(v, (int, float)) for v in stroke[0][:5])
            and all(isinstance(v, (int, float)) for v in stroke[1][:5])
        ):
            xs_seq = [float(v) for v in stroke[0]]
            ys_seq = [float(v) for v in stroke[1]]
        else:
            # Fallback: assume it's already a list of [x, y] points.
            xs_seq = []
            ys_seq = []
            for pt in stroke:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                xs_seq.append(float(pt[0]))
                ys_seq.append(float(pt[1]))

        n = min(len(xs_seq), len(ys_seq))
        if n <= 0:
            continue

        total_points += n
        xs.extend(xs_seq[:n])
        ys.extend(ys_seq[:n])

        # Corner counts per stroke
        total_corners += _count_corners_from_sequences(xs_seq[:n], ys_seq[:n], corner_angle_rad)

        # Segment directions + lengths
        if n < 2:
            continue
        for i in range(n - 1):
            dx = xs_seq[i + 1] - xs_seq[i]
            dy = ys_seq[i + 1] - ys_seq[i]
            length = math.hypot(dx, dy)
            if length == 0.0:
                continue
            segment_lengths.append(float(length))
            angles.append(math.atan2(dy, dx))

    if xs:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        bbox_width = float(maxx - minx)
        bbox_height = float(maxy - miny)
        bbox_diag = math.hypot(bbox_width, bbox_height)
    else:
        bbox_width = 0.0
        bbox_height = 0.0
        bbox_diag = 0.0

    ink_length_total = float(sum(segment_lengths)) if segment_lengths else 0.0
    num_segments = len(segment_lengths)
    ink_length_mean = ink_length_total / float(num_segments) if num_segments > 0 else 0.0
    ink_length_norm_diag = ink_length_total / (bbox_diag + eps)

    bbox_aspect_ratio = bbox_width / (bbox_height + eps) if bbox_height is not None else 0.0

    corners_count = float(total_corners)
    corners_density = corners_count / float(max(total_points - 2, 1))

    hist = _direction_histogram(angles, bins=direction_bins)
    if hist.shape[0] != 8:
        # Keep this stable: scripts/train_baseline.py assumes 8 bins (18 total dims).
        raise ValueError(f"Expected direction_bins=8 for 18-dim feature vector, got {hist.shape[0]}.")

    feats = np.asarray(
        [
            ink_length_total,
            ink_length_mean,
            float(num_strokes),
            float(total_points),
            float(bbox_width),
            float(bbox_height),
            float(bbox_aspect_ratio),
            float(ink_length_norm_diag),
            float(corners_count),
            float(corners_density),
        ],
        dtype=np.float32,
    )
    feats = np.concatenate([feats, hist.astype(np.float32)], axis=0)
    assert feats.shape == (18,), f"Unexpected feature shape: {feats.shape}"
    return feats


def _load_label_names_and_features(
    input_path: str,
    *,
    limit: Optional[int],
    include_unrecognized: bool,
    direction_bins: int,
    corner_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    label_to_name: Dict[int, str] = {}

    processed = 0
    with open(input_path, "r") as f:
        for line in f:
            rec = json.loads(line)

            if (not include_unrecognized) and (not rec.get("recognized", False)):
                continue

            label = int(rec["label"])
            name = str(rec["category"])
            label_to_name[label] = name

            feats = compute_stroke_features(
                rec.get("drawing", []),
                direction_bins=direction_bins,
                corner_angle_deg=corner_angle_deg,
            )
            features.append(feats)
            labels.append(label)

            processed += 1
            if limit is not None and processed >= limit:
                break

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # label_to_name may be incomplete when using --limit.
    # We still output a list indexed by the numeric label id so downstream
    # training code can reliably interpret y-values (even if some names are placeholders).
    discovered_labels = sorted(label_to_name.keys())
    if discovered_labels:
        max_label = max(discovered_labels)
        label_names = [label_to_name.get(i, f"label_{i}") for i in range(max_label + 1)]
    else:
        label_names = []

    config = {
        "input_path": input_path,
        "include_unrecognized": include_unrecognized,
        "direction_bins": direction_bins,
        "corner_angle_deg": corner_angle_deg,
        "feature_dim": int(X.shape[1]),
        "num_samples": int(X.shape[0]),
        "label_cardinality_observed": int(len(discovered_labels)),
        "observed_label_values": discovered_labels,
    }
    return X, y, label_names, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 18 stroke features from strokes.ndjson.")
    parser.add_argument(
        "--input",
        default="data/processed/strokes.ndjson",
        help="Path to strokes.ndjson",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Where to write X_features.npy, y_features.npy, and label_names.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records to process (after filtering).",
    )
    parser.add_argument(
        "--include-unrecognized",
        action="store_true",
        help="Include records where recognized=False",
    )
    parser.add_argument(
        "--direction-bins",
        type=int,
        default=8,
        help="Direction histogram bins (must be 8 for 18-dim output).",
    )
    parser.add_argument(
        "--corner-angle-deg",
        type=float,
        default=30.0,
        help="Turning angle threshold for corner counting.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, label_names, config = _load_label_names_and_features(
        args.input,
        limit=args.limit,
        include_unrecognized=args.include_unrecognized,
        direction_bins=args.direction_bins,
        corner_angle_deg=args.corner_angle_deg,
    )

    X_path = os.path.join(args.output_dir, "X_features.npy")
    y_path = os.path.join(args.output_dir, "y_features.npy")
    label_path = os.path.join(args.output_dir, "label_names.json")
    config_path = os.path.join(args.output_dir, "feature_config.json")

    np.save(X_path, X)
    np.save(y_path, y)
    # Keep label_names as a list (common for students); include config separately.
    with open(label_path, "w") as f:
        json.dump(label_names, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    feature_names = [
        "ink_length_total",
        "ink_length_mean_segment",
        "num_strokes",
        "num_points",
        "bbox_width",
        "bbox_height",
        "bbox_aspect_ratio",
        "ink_length_norm_diag",
        "corners_count",
        "corners_density",
    ] + [f"dir_hist_bin_{i}" for i in range(args.direction_bins)]
    # sanity check for 18-dim output
    assert len(feature_names) == X.shape[1], (len(feature_names), X.shape[1])

    with open(config_path, "r+") as f:
        cfg = json.load(f)
        cfg["feature_names"] = feature_names
        f.seek(0)
        json.dump(cfg, f, indent=2)
        f.truncate()

    print(f"Done. Wrote {X_path}, {y_path}, {label_path}")
    print(f"Shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()

