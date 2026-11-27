
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from team_audio import CLASSES, get_clip
from src.dsp_reference import features

TEAM = 11
WIN_MS = 30
HOP_MS = 20
NFFT = 512
MELS = 40

OUT = ROOT / "results/mission5"
DATASETS = OUT / "datasets"
REPORTS = OUT / "reports"
GRAPHS = OUT / "graphs"

CONFIGS = [
    ("int8_s1", 16000, True),
    ("int8_s025", 16000, True),
    ("8khz_no_filter", 8000, True),
]

summary = []

for name, fs, reference_filter in CONFIGS:

    print("\nProcessing:", name)

    path = DATASETS / f"{name}.npz"
    saved = np.load(path)

    board = saved["board_features"].astype(np.float32)
    labels = saved["labels"]
    clip_ids = saved["clip_id"].astype(int)
    first_samples = saved["first_sample"].astype(int)

    win = int(fs * WIN_MS / 1000)
    hop = int(fs * HOP_MS / 1000)

    # Calculate the correct laptop reference for each clip.
    reference_cache = {}

    for clip_id in np.unique(clip_ids):

        x = get_clip(
            TEAM,
            int(clip_id),
            fs=fs,
            antialias=reference_filter
        )

        reference_cache[int(clip_id)] = features(
            x,
            fs=fs,
            win=win,
            hop=hop,
            nfft=NFFT,
            mels=MELS,
            mfcc=0
        )

    reference = []

    for clip_id, start in zip(clip_ids, first_samples):

        local_start = int(start) - int(clip_id) * fs

        if local_start % hop != 0:
            raise ValueError("Unexpected frame alignment")

        frame_index = local_start // hop
        ref = reference_cache[int(clip_id)]

        if frame_index >= len(ref):
            raise ValueError("Reference frame out of range")

        reference.append(ref[frame_index])

    reference = np.asarray(reference, dtype=np.float32)

    if reference.shape != board.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, "
            f"board={board.shape}"
        )

    error = np.abs(reference - board)

    print("Frames:", len(error))
    print("Mean absolute error:", float(error.mean()), "dB")
    print("Worst absolute error:", float(error.max()), "dB")

    # Per-class, per-band report.
    report_path = REPORTS / f"{name}_corrected_errors.csv"

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "class", "mel_band",
            "mean_absolute_error_db",
            "maximum_absolute_error_db"
        ])

        heatmap = []

        for label in CLASSES:

            mask = labels == label
            class_error = error[mask]

            if len(class_error) == 0:
                raise ValueError(f"No frames found for {label}")

            heatmap.append(class_error.mean(axis=0))

            for band in range(MELS):
                writer.writerow([
                    name,
                    label,
                    band,
                    float(class_error[:, band].mean()),
                    float(class_error[:, band].max())
                ])

    heatmap = np.asarray(heatmap)

    plt.figure(figsize=(14, 5))
    plt.imshow(heatmap, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean absolute difference (dB)")
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Mel band")
    plt.ylabel("Class")
    plt.title(f"Corrected Board vs Laptop: {name}")
    plt.tight_layout()
    plt.savefig(
        GRAPHS / f"{name}_heatmap.png",
        dpi=200
    )
    plt.close()

    # Replace the laptop reference in the saved dataset.
    np.savez_compressed(
        path,
        board_features=board,
        laptop_features=reference,
        labels=labels,
        clip_id=clip_ids,
        first_sample=first_samples
    )

    summary.append({
        "configuration": name,
        "sampling_rate": fs,
        "reference_filter": reference_filter,
        "frames": len(error),
        "mean_error_db": float(error.mean()),
        "worst_error_db": float(error.max())
    })

    print("Saved corrected report:", report_path)
    print("Saved corrected heatmap:", GRAPHS / f"{name}_heatmap.png")

summary_path = REPORTS / "configuration_summary_corrected.csv"

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\n===== CORRECTED CONFIGURATION SUMMARY =====")
for row in summary:
    print(row)

print("\nSaved:", summary_path)
