
import os
import numpy as np
import matplotlib.pyplot as plt

from src.dsp_reference import features
from team_audio import get_clip, clip_list


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEAM = 11
FS = 16000
WIN = 480
HOP = 320
NFFT = 512
MELS = 40

CLASSES = ["on", "off", "stop", "go", "_unknown", "_silence"]

OUTPUT_DIR = "results/mission3/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Get available audio clips
# --------------------------------------------------

clips = clip_list(TEAM)

print(f"Total available clips: {len(clips)}")


# --------------------------------------------------
# 3. Find one clip belonging to each class
# --------------------------------------------------

def find_clip_for_class(label):

    for clip_id, clip_label in clips:
        if clip_label == label:
            return clip_id

    raise ValueError(f"No clip found for class: {label}")


# --------------------------------------------------
# 4. Load audio clip
# --------------------------------------------------

def load_clip(clip_id):

    x = get_clip(
        TEAM,
        clip_id,
        fs=FS,
        antialias=True
    )

    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------
# 5. Extract log-mel features
# --------------------------------------------------

def extract_logmel_features(x):

    mel = features(
        x,
        fs=FS,
        win=WIN,
        hop=HOP,
        nfft=NFFT,
        mels=MELS,
        mfcc=0
    )

    if mel.ndim != 2 or mel.shape[1] != MELS:
        raise ValueError(f"Unexpected feature shape: {mel.shape}")

    if not np.isfinite(mel).all():
        raise ValueError("Log-mel features contain NaN or Inf values.")

    return mel


# --------------------------------------------------
# 6. Select one clip per class and extract features
# --------------------------------------------------

class_features = {}

for label in CLASSES:

    print(f"\nProcessing class: {label}")

    # Find clip ID
    clip_id = find_clip_for_class(label)

    # Load audio
    x = load_clip(clip_id)

    # Extract log-mel features
    mel = extract_logmel_features(x)

    class_features[label] = (clip_id, mel)

    print(f"Clip ID: {clip_id}")
    print(f"Audio samples: {len(x)}")
    print(f"Feature shape: {mel.shape}")


# --------------------------------------------------
# 7. Determine common colour scale
# --------------------------------------------------

all_values = np.concatenate([
    mel.flatten()
    for _, mel in class_features.values()
])

vmin = np.percentile(all_values, 5)
vmax = np.percentile(all_values, 95)


# --------------------------------------------------
# 8. Plot and save log-mel spectrograms
# --------------------------------------------------

for label, (clip_id, mel) in class_features.items():

    plt.figure(figsize=(10, 4))

    time_axis = np.arange(mel.shape[0]) * HOP / FS

    plt.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        extent=[
            time_axis[0],
            time_axis[-1] + HOP / FS,
            0,
            MELS
        ],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    plt.title(f"Log-Mel Spectrogram - Class: {label}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency Bin")

    plt.colorbar(label="Log-Mel Energy (dB)")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"logmel_{label}.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 9. Completion
# --------------------------------------------------

print("\nAll six log-mel spectrograms generated successfully.")
