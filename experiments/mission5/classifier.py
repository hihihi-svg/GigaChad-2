
import os
import sys
import csv
import numpy as np

from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from team_audio import get_clip, CLASSES
from src.dsp_reference import features as extract_features

TEAM = 11
RANDOM_SEED = 11

REPORT_DIR = ROOT / "results" / "mission5" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load baseline Mission 4 board features and recreate
# the original float32 filtered laptop reference.
# --------------------------------------------------

def load_baseline():
    path = ROOT / "results" / "mission4" / "dataset" / "team11_features.npz"
    data = np.load(path)

    board = data["features"]
    labels = data["label"]
    clip_ids = data["clip_id"]

    laptop = np.zeros_like(board)

    for clip_id in np.unique(clip_ids):
        mask = clip_ids == clip_id
        audio = get_clip(
            TEAM,
            int(clip_id),
            fs=16000,
            antialias=True
        )
        reference = extract_features(audio, fs=16000)

        if len(reference) != np.sum(mask):
            raise ValueError(f"Frame mismatch for baseline clip {clip_id}")

        laptop[mask] = reference

    return {
        "board": board,
        "laptop": laptop,
        "labels": labels,
        "clip_ids": clip_ids
    }


# --------------------------------------------------
# Load additional Mission 5 configurations.
# laptop_features contains the corrected references.
# --------------------------------------------------

def load_configuration(name):
    path = ROOT / "results" / "mission5" / "datasets" / f"{name}.npz"
    data = np.load(path)

    return {
        "board": data["board_features"],
        "laptop": data["laptop_features"],
        "labels": data["labels"],
        "clip_ids": data["clip_id"]
    }


# --------------------------------------------------
# Convert frame-level features into one vector per clip:
# 40 means + 40 standard deviations = 80 dimensions.
# --------------------------------------------------

def aggregate_clips(data):
    output = {}

    for clip_id in np.unique(data["clip_ids"]):
        mask = data["clip_ids"] == clip_id

        labels = np.unique(data["labels"][mask])
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for clip {clip_id}")

        board_frames = data["board"][mask]
        laptop_frames = data["laptop"][mask]

        if len(board_frames) == 0:
            raise ValueError(f"No frames for clip {clip_id}")

        output[int(clip_id)] = {
            "label": str(labels[0]),
            "board": np.concatenate([
                board_frames.mean(axis=0),
                board_frames.std(axis=0)
            ]),
            "laptop": np.concatenate([
                laptop_frames.mean(axis=0),
                laptop_frames.std(axis=0)
            ])
        }

    return output


# --------------------------------------------------
# Create one shared, class-balanced split.
# 16 training clips + 8 testing clips per class.
# --------------------------------------------------

def create_split(data):
    rng = np.random.default_rng(RANDOM_SEED)

    split = {"train": [], "test": []}

    for label in CLASSES:
        ids = sorted([
            clip_id for clip_id, item in data.items()
            if item["label"] == label
        ])

        if len(ids) != 24:
            raise ValueError(
                f"Expected 24 clips for {label}, found {len(ids)}"
            )

        shuffled = rng.permutation(ids)
        split["train"].extend(shuffled[:16].tolist())
        split["test"].extend(shuffled[16:].tolist())

    return split


# --------------------------------------------------
# Standardise using training laptop features only.
# Train class centroids and predict by Euclidean distance.
# --------------------------------------------------

def fit_and_evaluate(data, split, test_source):
    train_ids = split["train"]
    test_ids = split["test"]

    X_train = np.array([
        data[i]["laptop"] for i in train_ids
    ])
    y_train = np.array([
        data[i]["label"] for i in train_ids
    ])

    X_test = np.array([
        data[i][test_source] for i in test_ids
    ])
    y_test = np.array([
        data[i]["label"] for i in test_ids
    ])

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    centroids = {}

    for label in CLASSES:
        centroids[label] = X_train[y_train == label].mean(axis=0)

    predictions = []

    for vector in X_test:
        distances = {
            label: np.linalg.norm(vector - centroids[label])
            for label in CLASSES
        }
        predictions.append(min(distances, key=distances.get))

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test) * 100

    per_class = {}

    for label in CLASSES:
        mask = y_test == label
        per_class[label] = (
            np.mean(predictions[mask] == y_test[mask]) * 100
        )

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for actual, predicted in zip(y_test, predictions):
        confusion[CLASSES.index(actual), CLASSES.index(predicted)] += 1

    return accuracy, per_class, confusion


# --------------------------------------------------
# Run all four configurations.
# --------------------------------------------------

configurations = {
    "int16_16khz_filtered": load_baseline(),
    "int8_s1": load_configuration("int8_s1"),
    "int8_s025": load_configuration("int8_s025"),
    "int16_8khz_no_filter": load_configuration("8khz_no_filter")
}

# One common split across all configurations.
first_data = aggregate_clips(configurations["int16_16khz_filtered"])
split = create_split(first_data)

# Save the exact clip split.
with open(REPORT_DIR / "classifier_clip_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id", "split"])
    for clip_id in split["train"]:
        writer.writerow([clip_id, "train"])
    for clip_id in split["test"]:
        writer.writerow([clip_id, "test"])

results = []

for name, raw_data in configurations.items():
    data = aggregate_clips(raw_data)

    print(f"\n===== {name} =====")

    for source in ["laptop", "board"]:
        accuracy, per_class, confusion = fit_and_evaluate(
            data, split, source
        )

        print(f"\nTest source: {source}")
        print(f"Overall accuracy: {accuracy:.2f}%")

        for label in CLASSES:
            print(f"  {label}: {per_class[label]:.2f}%")

        results.append({
            "configuration": name,
            "test_source": source,
            "accuracy_percent": accuracy,
            **{
                f"{label}_accuracy_percent": per_class[label]
                for label in CLASSES
            }
        })

        confusion_path = REPORT_DIR / f"{name}_{source}_confusion.csv"

        with open(confusion_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual/predicted"] + CLASSES)
            for i, label in enumerate(CLASSES):
                writer.writerow([label] + confusion[i].tolist())

# Save summary.
summary_path = REPORT_DIR / "classifier_results.csv"

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n===== FINAL CLASSIFICATION SUMMARY =====")
for row in results:
    print(
        f"{row['configuration']:25s} "
        f"{row['test_source']:8s} "
        f"{row['accuracy_percent']:.2f}%"
    )

print(f"\nSaved results: {summary_path}")
print(f"Saved confusion matrices in: {REPORT_DIR}")
