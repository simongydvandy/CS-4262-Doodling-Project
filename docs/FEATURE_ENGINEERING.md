# Feature Engineering — Classical Baseline

This document describes the 24-dimensional hand-engineered feature vector used by the classical baseline models. Features are extracted from the raw QuickDraw stroke sequences by `scripts/build_stroke_features.py`.

Each drawing is represented as a list of strokes, where each stroke is a sequence of (x, y) coordinates. No pixel rendering is used — all features are computed directly from the stroke geometry.

---

## Feature Families

### Family 1 — Counting (4 features)

These features capture the structural complexity of a drawing in terms of how many strokes it has and how densely sampled those strokes are.

| # | Name | Description |
|---|------|-------------|
| 0 | `num_strokes` | Number of separate pen-down strokes in the drawing |
| 1 | `num_points` | Total number of (x, y) coordinate pairs across all strokes |
| 2 | `avg_points_per_stroke` | Mean number of points per stroke (`num_points / num_strokes`) |
| 3 | `max_points_per_stroke` | Point count of the longest (most sampled) stroke |

**Rationale:** Simple drawings like a circle use 1–2 strokes; complex drawings like a face use many. The sampling density (points per stroke) reflects how much detail was captured per pen motion.

---

### Family 2 — Geometry (6 features)

These features describe the spatial extent and visual center of mass of the drawing, normalized to be position-independent.

| # | Name | Description |
|---|------|-------------|
| 4 | `bbox_width` | Width of the axis-aligned bounding box across all points |
| 5 | `bbox_height` | Height of the axis-aligned bounding box across all points |
| 6 | `bbox_aspect_ratio` | `bbox_width / bbox_height` — shape of the bounding box |
| 7 | `log_bbox_area` | `log1p(bbox_width × bbox_height)` — compressed bounding box area |
| 8 | `centroid_x_norm` | Horizontal center of mass, normalized to `[0, 1]` within the bounding box |
| 9 | `centroid_y_norm` | Vertical center of mass, normalized to `[0, 1]` within the bounding box |

**Rationale:** Aspect ratio distinguishes wide drawings (bus, bridge) from tall ones (tree, candle). The normalized centroid captures where the visual weight sits within the drawing — e.g. a mushroom is top-heavy, a wine glass is bottom-heavy.

**Design notes:**
- `log_bbox_area` compresses the wide range of bounding box areas (tiny doodles vs full-canvas drawings) so outliers don't dominate after StandardScaler.
- `centroid_x/y_norm` are normalized to bounding-box-relative coordinates so the feature encodes where within the drawing the visual mass sits, not where on the canvas the user happened to draw.

---

### Family 3 — Ink (3 features)

These features measure the total amount of pen movement in a drawing, capturing how "filled in" or "sketchy" it is.

| # | Name | Description |
|---|------|-------------|
| 10 | `ink_length_total` | Sum of all Euclidean segment lengths across all strokes |
| 11 | `ink_length_mean_segment` | Average length per segment (`ink_length_total / num_segments`) |
| 12 | `ink_length_norm_diag` | `ink_length_total` normalized by the bounding box diagonal |

**Rationale:** A circle has smooth, long ink segments; a scribble has many short ones. Normalizing by the bounding box diagonal (`ink_length_norm_diag`) removes size effects — a large circle and a small circle have similar normalized ink length.

---

### Family 4 — Direction (9 features)

These features encode the distribution of stroke directions across the drawing, capturing overall orientation and symmetry.

| # | Name | Description |
|---|------|-------------|
| 13–20 | `dir_hist_bin_0` … `dir_hist_bin_7` | 8-bin normalized histogram of segment angles over [0°, 360°) |
| 21 | `direction_entropy` | Shannon entropy of the direction histogram |

**How the histogram works:** Each segment between consecutive points has an angle computed via `atan2(dy, dx)`, mapped to [0°, 360°), and binned into one of 8 equal-width bins (45° each). The histogram is normalized so all 8 bins sum to 1.

**Rationale:**
- A circle has a uniform direction distribution (all 8 bins roughly equal) → high entropy
- A horizontal line has almost all weight in bins 0 and 4 → low entropy
- A star has concentrated peaks at specific angles → medium entropy with clear peaks

**Design note:** `dominant_direction` (the argmax bin index) was intentionally removed. The argmax index is a circular value — bin 7 (315–360°) is adjacent to bin 0 (0–45°) — but treating it as a continuous integer misleads the model into thinking they are far apart. The 8 histogram bins already encode this information without the encoding error.

---

### Family 5 — Curvature (2 features)

These features capture how angular or curved the strokes are.

| # | Name | Description |
|---|------|-------------|
| 22 | `corners_per_stroke` | Average number of sharp turns per stroke |
| 23 | `corners_density` | Corners per total point count |

**How corners are detected:** For each consecutive triple of points (prev, curr, next), the turning angle is computed using the dot product of the incoming and outgoing vectors. A point is counted as a corner if the turning angle exceeds 30°.

**Rationale:** Polygonal shapes (star, house, triangle) have high corner counts; smooth shapes (circle, oval, wave) have low corner counts. `corners_per_stroke` normalizes by stroke count so complex multi-stroke drawings are compared fairly against simple single-stroke ones.

---

## Summary Table

| # | Feature | Family | Unit / Range |
|---|---------|--------|-------------|
| 0 | `num_strokes` | Counting | integer ≥ 1 |
| 1 | `num_points` | Counting | integer ≥ 1 |
| 2 | `avg_points_per_stroke` | Counting | float > 0 |
| 3 | `max_points_per_stroke` | Counting | integer ≥ 1 |
| 4 | `bbox_width` | Geometry | pixels |
| 5 | `bbox_height` | Geometry | pixels |
| 6 | `bbox_aspect_ratio` | Geometry | float > 0 |
| 7 | `log_bbox_area` | Geometry | log-pixels² |
| 8 | `centroid_x_norm` | Geometry | [0, 1] |
| 9 | `centroid_y_norm` | Geometry | [0, 1] |
| 10 | `ink_length_total` | Ink | pixels |
| 11 | `ink_length_mean_segment` | Ink | pixels |
| 12 | `ink_length_norm_diag` | Ink | unitless |
| 13–20 | `dir_hist_bin_0..7` | Direction | [0, 1], sum = 1 |
| 21 | `direction_entropy` | Direction | [0, log(8)] |
| 22 | `corners_per_stroke` | Curvature | float ≥ 0 |
| 23 | `corners_density` | Curvature | [0, 1] |
