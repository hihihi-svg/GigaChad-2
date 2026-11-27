
import numpy as np

from board_link import Board
from team_audio import get_clip
from src.dsp_reference import features

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEAM = 11
PORT = "COM8"
BAUD = 921600

FS = 16000
WIN = 480
HOP = 320
NFFT = 512
MELS = 40

TOLERANCE_DB = 0.05

# --------------------------------------------------
# Load Team 11 audio
# --------------------------------------------------

print("Loading Team 11 audio...")

clips = [
    get_clip(TEAM, clip_id, fs=FS, antialias=True)
    for clip_id in range(10)
]

audio = np.concatenate(clips)

print("Audio samples:", len(audio))
print("Duration:", len(audio) / FS, "seconds")

# --------------------------------------------------
# Quantise to signed 16-bit PCM
# --------------------------------------------------

payload = (
    np.clip(audio, -1.0, 1.0) * 32767
).astype("<i2").tobytes()

# Reconstruct the exact quantised signal for NumPy.
quantised_audio = (
    np.frombuffer(payload, dtype="<i2")
    .astype(np.float32) / 32768.0
)

# --------------------------------------------------
# Calculate NumPy reference features
# --------------------------------------------------

print("\nCalculating NumPy reference features...")

numpy_features = features(
    quantised_audio,
    fs=FS,
    win=WIN,
    hop=HOP,
    nfft=NFFT,
    mels=MELS,
    mfcc=0
)

print("NumPy feature shape:", numpy_features.shape)

# --------------------------------------------------
# Get features from ESP32
# --------------------------------------------------

board = Board(PORT, BAUD)

try:
    print("\nStreaming audio to ESP32...")
    print("Port:", PORT)

    result = board.stream(
        payload=payload,
        fs=FS,
        fmt="i16",
        scale=1.0,
        win=WIN,
        hop=HOP,
        nfft=NFFT,
        mels=MELS,
        mfcc=0,
        feat=True,
        store="none",
        speed=1.0,
        samples_per_packet=320
    )

finally:
    board.close()

esp_features = result["features"]
first_samples = result["first_sample"]

print("\nESP32 feature shape:", esp_features.shape)
print("Expected frames:", result["expected_features"])
print("Received frames:", len(esp_features))
print("Lost frames:", result["lost_features"])

# --------------------------------------------------
# Align ESP32 frames with NumPy frames
# --------------------------------------------------

if len(esp_features) == 0:
    raise RuntimeError("ESP32 returned no feature frames.")

frame_indices = first_samples // HOP

if np.any(first_samples % HOP != 0):
    raise RuntimeError("Unexpected ESP32 frame alignment.")

if np.any(frame_indices >= len(numpy_features)):
    raise RuntimeError("ESP32 returned an invalid frame index.")

reference = numpy_features[frame_indices]

if reference.shape != esp_features.shape:
    raise RuntimeError(
        f"Shape mismatch: NumPy {reference.shape}, "
        f"ESP32 {esp_features.shape}"
    )

# --------------------------------------------------
# Calculate numerical error
# --------------------------------------------------

absolute_error = np.abs(reference - esp_features)

max_error = float(np.max(absolute_error))
mean_error = float(np.mean(absolute_error))

print("\n========== ACCURACY RESULTS ==========")

print("Maximum absolute error:", max_error, "dB")
print("Mean absolute error:", mean_error, "dB")
print("Required tolerance:", TOLERANCE_DB, "dB")

if result["lost_features"] != 0:
    print("\nINCOMPLETE: Some feature frames were lost.")
    print("Repeat the test after resolving frame loss.")

elif max_error <= TOLERANCE_DB:
    print("\nPASS: FFT accuracy is within 0.05 dB.")

else:
    print("\nFAIL: FFT accuracy exceeds 0.05 dB.")

print("\n======================================")
