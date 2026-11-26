### Before Mission 1 — USB Throughput Prediction

**Question asked:** How many bytes per second can be sent to an ESP32-S3 over USB?

**LLM response:**

The ESP32-S3 native USB interface supports USB Full-Speed at 12 Mbps. Dividing by 8 gives a theoretical raw transfer rate of 1,500,000 bytes/s (1.5 MB/s).

This is not the actual application payload throughput because USB protocol overhead, packet handling, host-side buffering and firmware processing reduce the effective rate. Therefore, the actual throughput must be measured experimentally.

For comparison, 16 kHz mono audio requires:
- float32: 64,000 bytes/s
- int16: 32,000 bytes/s
- int8: 16,000 bytes/s

**Prediction:** Native USB should have sufficient theoretical bandwidth for all three audio formats. Actual real-time feasibility will be established through Mission 1 measurements.

**Verification status:** Theoretical prediction only; experimental throughput measurement pending.

**Source:** Espressif ESP32-S3 Technical Reference Manual and USB-OTG documentation.