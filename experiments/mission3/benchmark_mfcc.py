
import numpy as np

from board_link import Board
from team_audio import get_clip


# Configuration
TEAM = 11
PORT = "COM8"
BAUD = 921600
FS = 16000
SPEED = 1.0
FEATURES = True
# --------------------------------------------------
# Load 10 seconds of audio
# --------------------------------------------------

print("Loading 10 seconds of Team 11 audio...")

clips = [
    get_clip(TEAM, clip_id, fs=FS, antialias=True)
    for clip_id in range(10)
]

audio = np.concatenate(clips)

print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / FS, "seconds")

# Quantise and pack as signed 16-bit PCM
payload = (
    np.clip(audio, -1.0, 1.0) * 32767
).astype("<i2").tobytes()

print("Payload bytes:", len(payload))

# --------------------------------------------------
# Connect to ESP32 and stream
# --------------------------------------------------

board = Board(PORT, BAUD)

try:
    print("\nStarting USB benchmark...")
    print(f"Speed: {SPEED}")
    print(f"Features: {'ON' if FEATURES else 'OFF'}")

    result = board.stream(
        payload=payload,
        fs=FS,
        fmt="i16",
        scale=1.0,
        win=400,
        hop=160,
        nfft=512,
        mels=40,
        mfcc=13,
        feat=FEATURES,
        store="none",
        speed=1.0,
        samples_per_packet=320
    )

    # --------------------------------------------------
    # Display benchmark results
    # --------------------------------------------------

    compute = result["compute_us"]
    lag = result["lag_s"]
    summary = result["board"] or {}

    print("\n========== USB BENCHMARK RESULTS ==========")

    print("Expected feature frames:", result["expected_features"])
    print("Received feature frames:", len(compute))
    print("Lost feature frames:", result["lost_features"])

    print("Audio bytes sent:", result["bytes_sent"])
    print(f"Bytes received from ESP32: {result['bytes_received']}")
    print("Packets sent:", result["packets_sent"])
    print("Transmission time:", round(result["send_seconds"], 3), "seconds")

    if len(compute) > 0:
        print("\nFeature computation:")
        print("Average compute time:", round(float(np.mean(compute)), 2), "us")
        print("Maximum compute time:", int(np.max(compute)), "us")
        print("Minimum compute time:", int(np.min(compute)), "us")

    if len(lag) > 0:
        print("\nStreaming lag:")
        print("Average lag:", round(float(np.mean(lag)), 6), "seconds")
        print("Maximum lag:", round(float(np.max(lag)), 6), "seconds")

    print("\nESP32 summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("Bad CRC packets:", result["bad_crc_back"])

    print("\n============================================")

finally:
    board.close()
