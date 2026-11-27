
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team_audio import clip_list, get_clip
from board_link import Board


# Configuration
TEAM = 11
FS = 16000
PORT = "COM8"
BAUD = 921600

WIN = 480
HOP = 320
NFFT = 512
MELS = 40
MFCC = 0
SAMPLES_PER_PACKET = 320
SPEED = 1.0


# Get all 144 clips in their original order
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


# Join all clips into one continuous recording
audio = np.concatenate(audio)


# Convert to int16 using the Mission 3 quantisation method
payload = (
    np.clip(audio, -1, 1) * 32767
).astype("<i2").tobytes()


# Display input information
print("TEAM:", TEAM)
print("Number of clips:", len(clips))
print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / FS, "seconds")
print("Audio bytes:", len(payload))
print("Expected packets:", len(audio) // SAMPLES_PER_PACKET)
print("Labels:", len(labels))
print("Payload prepared successfully.")


# Connect to ESP32 and stream all clips
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
        samples_per_packet=SAMPLES_PER_PACKET,
        skip=(10, 11, 200)
    )

    # Display stream results
    print("\nMISSION 4 - TASK 2 RESULTS")
    print("Packets sent:", result["packets_sent"])
    print("Expected feature frames:", result["expected_features"])
    print("Received feature frames:", len(result["features"]))
    print("Lost feature frames:", result["lost_features"])
    print("Bad CRC on return:", result["bad_crc_back"])

    # Display complete board accounting
    print("\nBOARD ACCOUNTING")

    for key, value in result["board"].items():
        print(f"{key}: {value}")

except Exception as e:
    print("\nSTREAMING ERROR:", e)
    raise

finally:
    board.close()
    print("\nPACKET LOSS VERIFICATION")
    print("Intentionally skipped packets:", (10, 11, 200))
    print("Sequence gaps detected:", result["board"]["seq_gaps"])
    print("Bad CRC:", result["board"]["bad_crc"])
    print("Samples received:", result["board"]["samples"])

    print("\nFEATURE FRAME POSITIONS")
    first_samples = result["first_sample"]
    frame_numbers = result["frame_numbers"]

    for i, (frame_no, sample_no) in enumerate(
        zip(frame_numbers, first_samples)
    ):
        if 8 <= frame_no <= 15 or 196 <= frame_no <= 204:
            print(
                f"Frame {frame_no}: "
                f"first_sample={sample_no}"
            )