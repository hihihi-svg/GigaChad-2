
from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "results" / "reports"
GRAPHS = ROOT / "results" / "graphs"

INPUT = REPORTS / "realtime_comparison_detailed.csv"

groups = defaultdict(list)

# Read detailed experiment results
with INPUT.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        send = float(row["send_seconds"])
        audio = float(row["audio_seconds"])

        row["near_realtime"] = send <= audio * 1.05

        expected_samples = round(
            float(row["sample_rate"]) * audio
        )

        errors = sum(
            int(float(row[field]))
            for field in [
                "bad_crc", "bad_len", "seq_gaps",
                "duplicates", "uart_errors"
            ]
        )

        row["integrity_pass"] = (
            errors == 0
            and int(float(row["samples_received"])) == expected_samples
        )

        key = (
            row["interface"],
            float(row["baud_setting"]),
            float(row["sample_rate"]),
            row["format"]
        )

        groups[key].append(row)

# Build grouped summary
summary = []

for key, rows in groups.items():
    throughputs = [
        float(r["throughput_Bps"]) for r in rows
    ]

    durations = [
        float(r["send_seconds"]) for r in rows
    ]

    summary.append({
        "interface": key[0],
        "baud_setting": key[1],
        "sample_rate": key[2],
        "format": key[3],
        "tests": len(rows),
        "average_throughput_Bps": round(
            sum(throughputs) / len(throughputs), 2
        ),
        "minimum_throughput_Bps": round(min(throughputs), 2),
        "maximum_throughput_Bps": round(max(throughputs), 2),
        "maximum_send_seconds": round(max(durations), 4),
        "near_realtime_passes": sum(
            r["near_realtime"] for r in rows
        ),
        "integrity_passes": sum(
            r["integrity_pass"] for r in rows
        )
    })

# Save summary CSV
summary_path = REPORTS / "near_realtime_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

# Prepare throughput comparison graph
interfaces = [
    ("Native USB", 115200.0),
    ("UART", 115200.0),
    ("UART", 921600.0),
    ("UART", 2000000.0)
]

interface_labels = [
    "Native USB",
    "UART 115200",
    "UART 921600",
    "UART 2000000"
]

formats = [
    (16000.0, "f32"),
    (16000.0, "i8"),
    (8000.0, "i16"),
    (8000.0, "i8")
]

format_labels = [
    "16 kHz float32",
    "16 kHz int8",
    "8 kHz int16",
    "8 kHz int8"
]

values = []

for interface, baud in interfaces:
    interface_values = []

    for fs, fmt in formats:
        matching = [
            row for row in summary
            if row["interface"] == interface
            and row["baud_setting"] == baud
            and row["sample_rate"] == fs
            and row["format"] == fmt
        ]

        if matching:
            interface_values.append(
                matching[0]["average_throughput_Bps"]
            )
        else:
            interface_values.append(0)

    values.append(interface_values)

x = np.arange(len(formats))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, label in enumerate(interface_labels):
    ax.bar(
        x + (i - 1.5) * width,
        [values[i][j] for j in range(len(formats))],
        width,
        label=label
    )

ax.set_xlabel("Audio format")
ax.set_ylabel("Average payload throughput (bytes/second)")
ax.set_title("Real-Time Audio Transfer Throughput Comparison")
ax.set_xticks(x)
ax.set_xticklabels(format_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

graph_path = GRAPHS / "realtime_throughput_comparison.png"
fig.savefig(graph_path, dpi=300)
plt.close(fig)

# Print final summary
print("MISSION 1 NEAR-REAL-TIME ANALYSIS")
print("--------------------------------")

print("Total tests:", sum(r["tests"] for r in summary))
print(
    "Near-real-time passes:",
    sum(r["near_realtime_passes"] for r in summary)
)
print(
    "Integrity passes:",
    sum(r["integrity_passes"] for r in summary)
)

print("\nSaved files:")
print(summary_path)
print(graph_path)
