
from pathlib import Path
import sys
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from board_link import Board
import team_audio


# Experiment configuration
PORT = "COM9"
BAUD = 115200
TEAM = 11
NUM_CLIPS = 1

PACKET_SIZES = [1024, 320, 64, 8]
TRIALS = 1
SPEED = 1

# Remaining real-time formats
FORMATS = [
    (16000, "f32"),
    (16000, "i8"),
    (8000, "i16"),
    (8000, "i8"),
]


def prepare_audio(fs, fmt):
    """Generate audio and encode it in the selected wire format."""

    audio = team_audio.get_clip(
        TEAM, 0, fs=fs, antialias=True
    )

    if fmt == "f32":
        payload = audio.astype("<f4").tobytes()
    elif fmt == "i16":
        payload = (
            np.clip(audio, -1, 1) * 32767
        ).round().astype("<i2").tobytes()
    elif fmt == "i8":
        payload = (
            np.clip(audio, -1, 1) * 127
        ).round().astype("i1").tobytes()
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return audio, payload


def run_experiment(fs, fmt, payload):
    """Measure real-time transfer for one audio format."""

    results = []
    board = Board(PORT, baud=BAUD)

    try:
        print("\nBoard information:")
        print(board.info())

        for packet_size in PACKET_SIZES:
            print(f"\n{'=' * 45}")
            print(f"Format: {fmt} | Sampling rate: {fs} Hz")
            print(f"Packet size: {packet_size}")
            print(f"{'=' * 45}")

            for trial in range(1, TRIALS + 1):
                r = board.stream(
                    payload,
                    fs=fs,
                    fmt=fmt,
                    scale=1.0,
                    feat=False,
                    store="none",
                    speed=SPEED,
                    samples_per_packet=packet_size
                )

                summary = r["board"]
                throughput = (
                    r["bytes_sent"] / r["send_seconds"]
                )

                row = {
                    "interface": "UART",
                    "port": PORT,
                    "baud_setting": BAUD,
                    "sample_rate": fs,
                    "format": fmt,
                    "packet_size_samples": packet_size,
                    "trial": trial,
                    "payload_bytes": r["bytes_sent"],
                    "send_seconds": r["send_seconds"],
                    "audio_seconds": r["audio_seconds"],
                    "throughput_Bps": throughput,
                    "packets_sent": r["packets_sent"],
                    "frames_ok": summary.get("frames_ok"),
                    "samples_received": summary.get("samples"),
                    "bad_crc": summary.get("bad_crc"),
                    "bad_len": summary.get("bad_len"),
                    "seq_gaps": summary.get("seq_gaps"),
                    "duplicates": summary.get("dup"),
                    "uart_errors": summary.get("uart_errors"),
                    "completion_reason": summary.get("why")
                }

                results.append(row)

                print(f"\nTrial {trial}/{TRIALS}")
                print("Payload:", r["bytes_sent"], "bytes")
                print("Send time:", round(r["send_seconds"], 6))
                print("Throughput:", round(throughput, 2), "B/s")
                print("Board summary:", summary)

    finally:
        board.close()

    return results


def save_results(results):
    report_dir = ROOT / "results" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output = report_dir / "hardware_measurements.csv"

    fieldnames = list(results[0].keys())

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", output)


if __name__ == "__main__":
    all_results = []

    for fs, fmt in FORMATS:
        audio, payload = prepare_audio(fs, fmt)

        print("\n" + "=" * 55)
        print("FORMAT:", fmt)
        print("SAMPLING RATE:", fs)
        print("Audio samples:", len(audio))
        print("Audio duration:", len(audio) / fs, "seconds")
        print("Payload size:", len(payload), "bytes")

        results = run_experiment(fs, fmt, payload)
        all_results.extend(results)

    save_results(all_results)
    print("\nReal-time format experiments completed.")
