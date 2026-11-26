
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "results" / "reports"

FILES = [
    "usb_realtime_formats.csv",
    "uart_115200_realtime_formats.csv",
    "uart_921600_realtime_formats.csv",
    "uart_2000000_realtime_formats.csv",
]

numeric_fields = [
    "baud_setting", "sample_rate", "packet_size_samples",
    "send_seconds", "audio_seconds", "throughput_Bps",
    "samples_received", "bad_crc", "bad_len",
    "seq_gaps", "duplicates", "uart_errors"
]

details = []
groups = defaultdict(list)

for filename in FILES:
    path = REPORTS / filename

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            for field in numeric_fields:
                if field in row and row[field] != "":
                    row[field] = float(row[field])

            send_time = row["send_seconds"]
            audio_time = row["audio_seconds"]

            row["realtime_pass"] = send_time <= audio_time

            expected_samples = round(
                row["sample_rate"] * audio_time
            )

            errors = sum(
                row.get(field, 0) or 0
                for field in [
                    "bad_crc", "bad_len", "seq_gaps",
                    "duplicates", "uart_errors"
                ]
            )

            row["integrity_pass"] = (
                errors == 0
                and row["samples_received"] == expected_samples
            )

            details.append(row)

            key = (
                row["interface"],
                row["baud_setting"],
                row["sample_rate"],
                row["format"]
            )
            groups[key].append(row)

# Save detailed results
detail_path = REPORTS / "realtime_comparison_detailed.csv"

with detail_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=details[0].keys())
    writer.writeheader()
    writer.writerows(details)

# Generate grouped summary
summary = []

for key, rows in groups.items():
    throughputs = [r["throughput_Bps"] for r in rows]
    send_times = [r["send_seconds"] for r in rows]

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
        "maximum_send_seconds": round(max(send_times), 4),
        "realtime_passes": sum(r["realtime_pass"] for r in rows),
        "integrity_passes": sum(r["integrity_pass"] for r in rows)
    })

summary_path = REPORTS / "realtime_comparison_summary.csv"

with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("DETAILED RESULTS")
print(f"Total tests: {len(details)}")
print(f"Real-time passes: {sum(r['realtime_pass'] for r in details)}")
print(f"Integrity passes: {sum(r['integrity_pass'] for r in details)}")

print("\nGROUPED COMPARISON")
for row in summary:
    print(
        f"{row['interface']} | {row['baud_setting']} baud | "
        f"{row['sample_rate']} Hz {row['format']} | "
        f"Average: {row['average_throughput_Bps']} B/s | "
        f"Real-time: {row['realtime_passes']}/{row['tests']} | "
        f"Integrity: {row['integrity_passes']}/{row['tests']}"
    )

print("\nSaved:")
print(detail_path)
print(summary_path)
