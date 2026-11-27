
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from team_audio import CLASSES, clip_list, get_clip
from board_link import Board
from src.dsp_reference import features

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEAM = 11
PORT = "COM9"
BAUD = 921600
SPEED = 14.0

WIN_MS = 30
HOP_MS = 20
NFFT = 512
MELS = 40

OUT = ROOT / "results/mission5"
GRAPHS = OUT / "graphs"
REPORTS = OUT / "reports"
DATASETS = OUT / "datasets"

for folder in [GRAPHS, REPORTS, DATASETS]:
    folder.mkdir(parents=True, exist_ok=True)

clips = clip_list(TEAM)

# Each configuration:
# name, sampling rate, board format, scale,
# board input anti-alias filter, laptop reference anti-alias filter
CONFIGS = [
    ("int8_s1", 16000, "i8", 1.0, True, True),
    ("int8_s025", 16000, "i8", 0.25, True, True),
    ("8khz_no_filter", 8000, "i16", 1.0, False, True),
]


def quantise_audio(x, fmt, scale):
    """Quantise audio using the assignment's scale-based formula."""

    if fmt == "i8":
        q = np.clip(
            np.rint(x / scale * 127),
            -128, 127
        ).astype(np.int8)

        reconstructed = q.astype(np.float32) * scale / 127.0

    elif fmt == "i16":
        q = np.clip(
            np.rint(x * 32767),
            -32768, 32767
        ).astype("<i2")

        reconstructed = q.astype(np.float32) / 32768.0

    else:
        raise ValueError(fmt)

    return q.tobytes(), reconstructed


def run_configuration(config):
    name, fs, fmt, scale, board_filter, laptop_filter = config

    print("\n" + "=" * 60)
    print("CONFIGURATION:", name)
    print("=" * 60)

    win = int(fs * WIN_MS / 1000)
    hop = int(fs * HOP_MS / 1000)

    # Prepare all 144 clips.
    board_audio = []
    reference_audio = []
    clip_samples = fs

    for clip_id, label in clips:

        x_board = get_clip(
            TEAM, clip_id,
            fs=fs,
            antialias=board_filter
        )

        x_reference = get_clip(
            TEAM, clip_id,
            fs=fs,
            antialias=laptop_filter
        )

        board_audio.append(x_board)
        reference_audio.append(x_reference)

    # Quantise each board input and retain reconstructed samples
    # for the corresponding laptop reference.
    packed_clips = []
    reconstructed_clips = []

    for x in board_audio:
        payload, reconstructed = quantise_audio(
            x, fmt, scale
        )
        packed_clips.append(payload)
        reconstructed_clips.append(reconstructed)

    payload = b"".join(packed_clips)

    print("Sampling rate:", fs)
    print("Format:", fmt)
    print("Scale:", scale)
    print("Board anti-alias:", board_filter)
    print("Laptop reference anti-alias:", laptop_filter)
    print("Audio duration:", len(payload) / (fs * (1 if fmt == "i8" else 2)), "seconds")
    print("Payload bytes:", len(payload))

    # Stream to ESP32.
    board = Board(PORT, baud=BAUD)

    try:
        result = board.stream(
            payload=payload,
            fs=fs,
            fmt=fmt,
            scale=scale,
            win=win,
            hop=hop,
            nfft=NFFT,
            mels=MELS,
            mfcc=0,
            feat=True,
            store="none",
            speed=SPEED,
            samples_per_packet=320
        )
    finally:
        board.close()

    board_features = np.asarray(
        result["features"], dtype=np.float32
    )
    first_samples = np.asarray(
        result["first_sample"], dtype=np.int64
    )

    print("\nSTREAM RESULTS")
    print("Packets sent:", result["packets_sent"])
    print("Expected features:", result["expected_features"])
    print("Received features:", len(board_features))
    print("Lost features:", result["lost_features"])
    print("Bad CRC on return:", result["bad_crc_back"])
    print("Board summary:", result["board"])

    if result["lost_features"] != 0:
        raise RuntimeError(
            f"{name}: feature frames were lost. "
            "Do not use incomplete data for comparison."
        )

    if len(board_features) != len(first_samples):
        raise RuntimeError("Feature/sample alignment mismatch.")

    # Compare board features with the laptop reference.
    errors_by_class = {label: [] for label in CLASSES}
    valid_board = []
    valid_reference = []
    valid_labels = []
    valid_clip_ids = []
    valid_starts = []

    for i, start in enumerate(first_samples):

        start = int(start)
        clip_id = start // clip_samples

        if clip_id < 0 or clip_id >= len(clips):
            continue

        local_start = start - clip_id * clip_samples

        # Exclude frames crossing clip boundaries.
        if local_start < 0 or local_start + win > clip_samples:
            continue

        if local_start % hop != 0:
            raise RuntimeError(
                f"Unexpected frame alignment: {local_start}"
            )

        frame_index = local_start // hop

        ref = features(
            reconstructed_clips[clip_id],
            fs=fs,
            win=win,
            hop=hop,
            nfft=NFFT,
            mels=MELS,
            mfcc=0
        )

        if frame_index >= len(ref):
            raise RuntimeError("Reference frame index out of range.")

        reference_frame = ref[frame_index]
        board_frame = board_features[i]

        error = np.abs(reference_frame - board_frame)
        label = clips[clip_id][1]

        errors_by_class[label].append(error)

        valid_board.append(board_frame)
        valid_reference.append(reference_frame)
        valid_labels.append(label)
        valid_clip_ids.append(clip_id)
        valid_starts.append(start)

    # Save comparison data.
    valid_board = np.asarray(valid_board, dtype=np.float32)
    valid_reference = np.asarray(valid_reference, dtype=np.float32)
    valid_labels = np.asarray(valid_labels)
    valid_clip_ids = np.asarray(valid_clip_ids, dtype=np.int32)
    valid_starts = np.asarray(valid_starts, dtype=np.int64)

    all_errors = np.abs(valid_reference - valid_board)

    print("\nCOMPARISON RESULTS")
    print("Valid labelled frames:", len(all_errors))
    print("Mean absolute error:", float(np.mean(all_errors)), "dB")
    print("Worst absolute error:", float(np.max(all_errors)), "dB")

    # Per-class and per-band report.
    csv_path = REPORTS / f"{name}_band_class_errors.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "class", "mel_band",
            "mean_absolute_error_db",
            "maximum_absolute_error_db"
        ])

        for label in CLASSES:
            class_errors = np.asarray(errors_by_class[label])

            for band in range(MELS):
                writer.writerow([
                    name,
                    label,
                    band,
                    float(np.mean(class_errors[:, band])),
                    float(np.max(class_errors[:, band]))
                ])

    # Heatmap.
    matrix = []

    for label in CLASSES:
        matrix.append(
            np.mean(np.asarray(errors_by_class[label]), axis=0)
        )

    matrix = np.asarray(matrix)

    plt.figure(figsize=(14, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean absolute difference (dB)")
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Mel band")
    plt.ylabel("Class")
    plt.title(f"Board vs Laptop: {name}")
    plt.tight_layout()
    plt.savefig(GRAPHS / f"{name}_heatmap.png", dpi=200)
    plt.close()

    # Preserve features for classifier evaluation in Task 4.
    np.savez_compressed(
        DATASETS / f"{name}.npz",
        board_features=valid_board,
        laptop_features=valid_reference,
        labels=valid_labels,
        clip_id=valid_clip_ids,
        first_sample=valid_starts
    )

    print("\nSaved:", csv_path)
    print("Saved heatmap:", GRAPHS / f"{name}_heatmap.png")
    print("Saved dataset:", DATASETS / f"{name}.npz")

    return {
        "configuration": name,
        "sampling_rate": fs,
        "format": fmt,
        "scale": scale,
        "frames": len(all_errors),
        "mean_error_db": float(np.mean(all_errors)),
        "worst_error_db": float(np.max(all_errors)),
    }


# --------------------------------------------------
# RUN ALL THREE CONFIGURATIONS
# --------------------------------------------------

summary = []

for config in CONFIGS:
    summary.append(run_configuration(config))

summary_path = REPORTS / "configuration_summary.csv"

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\n" + "=" * 60)
print("MISSION 5 - TASK 3 SUMMARY")
print("=" * 60)

for row in summary:
    print(row)

print("\nSummary saved:", summary_path)
print("ALL CONFIGURATIONS COMPLETED")
