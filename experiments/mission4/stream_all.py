
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team_audio import clip_list, get_clip
from board_link import Board


# ============================================================
# CONFIGURATION
# ============================================================

TEAM = 11
FS = 16000
PORT = "COM9"
BAUD = 921600

WIN = 480
HOP = 320
NFFT = 512
MELS = 40
MFCC = 0

SAMPLES_PER_PACKET = 320
SPEED = 14.0

CLIP_SAMPLES = FS  # Each clip is 1 second


# ============================================================
# LOAD ALL 144 AUDIO CLIPS
# ============================================================

clips = clip_list(TEAM)

audio = []
labels = []

for clip_id, label in clips:

    x = get_clip(
        team=TEAM,
        clip_id=clip_id,
        fs=FS,
        antialias=True
    )

    audio.append(x)
    labels.append(label)


# ============================================================
# COMBINE CLIPS INTO ONE CONTINUOUS RECORDING
# ============================================================

audio = np.concatenate(audio)

# Convert to int16
payload = (
    np.clip(audio, -1, 1) * 32767
).astype("<i2").tobytes()


# ============================================================
# DISPLAY INPUT INFORMATION
# ============================================================

print("TEAM:", TEAM)
print("Number of clips:", len(clips))
print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / FS, "seconds")
print("Audio bytes:", len(payload))
print("Expected packets:", len(audio) // SAMPLES_PER_PACKET)
print("Labels:", len(labels))
print("Payload prepared successfully.")


# ============================================================
# CONNECT TO ESP32 AND STREAM ALL CLIPS
# ============================================================

board = Board(PORT, baud=BAUD)

try:

    print("\nStarting full 144-clip stream...")

    result = board.stream(
        payload,
        fs=FS,
        fmt="i16",
        scale=1.0,
        win=WIN,
        hop=HOP,
        nfft=NFFT,
        mels=MELS,
        mfcc=MFCC,
        feat=True,
        store="psram",
        max_bytes=len(payload),
        speed=SPEED,
        samples_per_packet=SAMPLES_PER_PACKET
    )

    # ========================================================
    # STREAM SPEED ANALYSIS
    # ========================================================

    actual_send_time = result["send_seconds"]
    audio_duration = result["audio_seconds"]

    effective_speed = (
        audio_duration / actual_send_time
        if actual_send_time > 0 else 0
    )

    print("\nSTREAM SPEED ANALYSIS")
    print("Requested speed:", SPEED, "x")
    print("Audio duration:", audio_duration, "seconds")
    print("Actual send time:", actual_send_time, "seconds")
    print("Effective speed:", round(effective_speed, 2), "x")

    # ========================================================
    # MISSION 4 - STREAM RESULTS
    # ========================================================

    print("\nMISSION 4 - STREAMING RESULTS")

    print("Packets sent:", result["packets_sent"])
    print("Expected feature frames:", result["expected_features"])
    print("Received feature frames:", len(result["features"]))
    print("Lost feature frames:", result["lost_features"])
    print("Bad CRC on return:", result["bad_crc_back"])

    print("\nBOARD ACCOUNTING")

    for key, value in result["board"].items():
        print(f"{key}: {value}")

    # ========================================================
    # MISSION 4 - TASK 6: LABEL FEATURE FRAMES
    # ========================================================

    print("\nMISSION 4 - TASK 6: FRAME LABELLING")

    features = np.asarray(
        result["features"],
        dtype=np.float32
    )

    first_samples = np.asarray(
        result["first_sample"],
        dtype=np.int64
    )

    # Verify feature and sample-position alignment
    if len(features) != len(first_samples):
        raise ValueError(
            "Feature count and first_sample count do not match!"
        )

    # Map clip IDs to their corresponding labels
    clip_label_map = {
        int(clip_id): label
        for clip_id, label in clips
    }

    valid_features = []
    frame_labels = []
    frame_clip_ids = []
    valid_first_samples = []

    discarded_boundary_frames = 0
    discarded_invalid_frames = 0

    # Process every feature frame
    for i, start in enumerate(first_samples):

        start = int(start)
        end = start + WIN

        # Reject invalid sample positions
        if start < 0:
            discarded_invalid_frames += 1
            continue

        # Identify the clip containing the frame's start
        clip_id = start // CLIP_SAMPLES

        if clip_id not in clip_label_map:
            discarded_invalid_frames += 1
            continue

        clip_start = clip_id * CLIP_SAMPLES
        clip_end = clip_start + CLIP_SAMPLES

        # Discard frames crossing clip boundaries
        if start < clip_start or end > clip_end:
            discarded_boundary_frames += 1
            continue

        # Append all associated values together
        valid_features.append(features[i])
        frame_labels.append(clip_label_map[clip_id])
        frame_clip_ids.append(clip_id)
        valid_first_samples.append(start)

    # Convert lists into NumPy arrays
    valid_features = np.asarray(
        valid_features,
        dtype=np.float32
    )

    frame_labels = np.asarray(frame_labels)
    frame_clip_ids = np.asarray(
        frame_clip_ids,
        dtype=np.int32
    )

    valid_first_samples = np.asarray(
        valid_first_samples,
        dtype=np.int64
    )

    # ========================================================
    # VERIFY DATASET
    # ========================================================

    print("\nFRAME LABELLING RESULTS")

    print("Original feature frames:", len(features))
    print("Valid labelled frames:", len(valid_features))
    print("Discarded boundary frames:", discarded_boundary_frames)
    print("Discarded invalid frames:", discarded_invalid_frames)

    print("\nARRAY SHAPES")

    print("Features:", valid_features.shape)
    print("Labels:", frame_labels.shape)
    print("Clip IDs:", frame_clip_ids.shape)
    print("First samples:", valid_first_samples.shape)

    # Check all arrays have matching lengths
    assert len(valid_features) == len(frame_labels)
    assert len(valid_features) == len(frame_clip_ids)
    assert len(valid_features) == len(valid_first_samples)

    # ========================================================
    # FRAMES PER CLASS
    # ========================================================

    print("\nFRAMES PER CLASS")

    for label in sorted(set(frame_labels.tolist())):

        count = int(np.sum(frame_labels == label))

        print(label, ":", count)

    # ========================================================
    # SAVE LABELLED DATASET
    # ========================================================

    output_dir = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "mission4"
        / "dataset"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / "team11_features.npz"

    np.savez_compressed(
        output_file,
        features=valid_features,
        label=frame_labels,
        clip_id=frame_clip_ids,
        first_sample=valid_first_samples
    )

    print("\nDATASET SAVED SUCCESSFULLY")
    print("File:", output_file)
    print("Total labelled frames:", len(valid_features))


except Exception as e:

    print("\nSTREAMING ERROR:", e)
    raise

finally:

    board.close()
