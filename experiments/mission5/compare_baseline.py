
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from team_audio import CLASSES, clip_list, get_clip
from src.dsp_reference import features

TEAM = 11
FS = 16000
WIN = 480
HOP = 320
NFFT = 512
MELS = 40

DATASET = ROOT / "results/mission4/dataset/team11_features.npz"
OUT = ROOT / "results/mission5"

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "graphs").mkdir(exist_ok=True)
(OUT / "reports").mkdir(exist_ok=True)

# Load the board's labelled Mission 4 features.
data = np.load(DATASET)

board_features = data["features"].astype(np.float32)
clip_ids = data["clip_id"].astype(int)
first_samples = data["first_sample"].astype(int)

clips = clip_list(TEAM)
label_map = dict(clips)

original_errors = []
quantised_errors = []
original_rows = []
quantised_rows = []

for clip_id, label in clips:

    # Recreate the original float32 clip.
    x = get_clip(
        TEAM, clip_id,
        fs=FS,
        antialias=True
    )

    # Reproduce the exact int16 representation sent to the board.
    payload = (
        np.clip(x, -1.0, 1.0) * 32767
    ).astype("<i2")

    quantised_x = payload.astype(np.float32) / 32768.0

    # Laptop features from original float32 audio.
    laptop_original = features(
        x, FS, WIN, HOP, NFFT, MELS, 0
    )

    # Laptop features from the same quantised samples as the board.
    laptop_quantised = features(
        quantised_x, FS, WIN, HOP, NFFT, MELS, 0
    )

    # Select the corresponding board frames for this clip.
    mask = clip_ids == clip_id

    board_clip = board_features[mask]
    starts = first_samples[mask]

    local_starts = starts - clip_id * FS

    if np.any(local_starts % HOP != 0):
        raise ValueError(
            f"Unexpected frame alignment in clip {clip_id}"
        )

    frame_indices = local_starts // HOP

    if np.any(frame_indices >= len(laptop_original)):
        raise ValueError(
            f"Invalid frame index in clip {clip_id}"
        )

    ref_original = laptop_original[frame_indices]
    ref_quantised = laptop_quantised[frame_indices]

    if board_clip.shape != ref_original.shape:
        raise ValueError(
            f"Shape mismatch in clip {clip_id}: "
            f"board={board_clip.shape}, "
            f"laptop={ref_original.shape}"
        )

    # Absolute errors in dB.
    err_original = np.abs(ref_original - board_clip)
    err_quantised = np.abs(ref_quantised - board_clip)

    original_errors.append(err_original)
    quantised_errors.append(err_quantised)

    original_rows.append((label, err_original))
    quantised_rows.append((label, err_quantised))

    print(
        f"Clip {clip_id:03d} | {label:10s} | "
        f"frames={len(board_clip)}"
    )

# Combine all frames.
original_errors = np.concatenate(original_errors, axis=0)
quantised_errors = np.concatenate(quantised_errors, axis=0)

print("\n===== BASELINE COMPARISON =====")
print("Total compared frames:", len(original_errors))
print("Original float32 mean error:",
      np.mean(original_errors), "dB")
print("Original float32 worst error:",
      np.max(original_errors), "dB")
print("Same int16 input mean error:",
      np.mean(quantised_errors), "dB")
print("Same int16 input worst error:",
      np.max(quantised_errors), "dB")

# Per-class, per-band statistics.
csv_path = OUT / "reports/baseline_band_class_errors.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "comparison", "class", "mel_band",
        "mean_absolute_error_db",
        "maximum_absolute_error_db"
    ])

    for comparison, rows in [
        ("original_float32", original_rows),
        ("same_int16_input", quantised_rows)
    ]:
        for label in CLASSES:
            class_errors = [
                error for row_label, error in rows
                if row_label == label
            ]

            errors = np.concatenate(class_errors, axis=0)

            for band in range(MELS):
                writer.writerow([
                    comparison,
                    label,
                    band,
                    float(np.mean(errors[:, band])),
                    float(np.max(errors[:, band]))
                ])

# Heatmaps: rows are classes, columns are mel bands.
for comparison, rows in [
    ("original_float32", original_rows),
    ("same_int16_input", quantised_rows)
]:
    matrix = []

    for label in CLASSES:
        errors = np.concatenate([
            e for row_label, e in rows
            if row_label == label
        ], axis=0)

        matrix.append(np.mean(errors, axis=0))

    matrix = np.asarray(matrix)

    plt.figure(figsize=(14, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean absolute difference (dB)")
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Mel band")
    plt.ylabel("Class")
    plt.title(f"Board vs Laptop: {comparison}")
    plt.tight_layout()

    plt.savefig(
        OUT / f"graphs/{comparison}_heatmap.png",
        dpi=200
    )
    plt.close()

print("\nSaved:")
print(csv_path)
print(OUT / "graphs/original_float32_heatmap.png")
print(OUT / "graphs/same_int16_input_heatmap.png")
print("\nMISSION 5 BASELINE COMPARISON COMPLETE")
