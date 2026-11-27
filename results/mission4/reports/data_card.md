# Team 11 – Mission 4 Data Card

## 1. Dataset Source
The dataset was generated from 144 audio clips provided through `team_audio.py`.
It contains six classes: on, off, stop, go, _unknown and _silence, with 24 clips per class.

## 2. Audio Processing
- Sampling frequency: 16,000 Hz
- Clip duration: 1 second
- Audio format: Signed 16-bit PCM
- Anti-alias filtering: Enabled
- Total audio samples: 2,304,000
- Total audio payload: 4,608,000 bytes

## 3. Feature Extraction
Features were extracted on the ESP32-S3 using the optimised FFT implementation.
- Window size: 480 samples (30 ms)
- Hop size: 320 samples (20 ms)
- FFT size: 512
- Feature type: Log-mel
- Mel coefficients: 40

## 4. Final Dataset
- Total labelled frames: 7,056
- Feature array shape: (7056, 40)
- Frames per class: 1,176
- Classes: 6
- Boundary-crossing frames discarded: 143

Each feature frame is stored with its class label, source clip ID and first sample position.

## 5. Data Loss and Verification
The final dataset was generated using COM9 at a requested speed of 14x.
The observed effective streaming speed was approximately 2.77x due to serial transmission limits.

All 7,200 packets were sent and received successfully:
- Sequence gaps: 0
- Bad CRC: 0
- Lost feature frames: 0

An additional intentional packet-loss test skipped packets 10, 11 and 200.
The board detected exactly three sequence gaps, confirming that packet loss was accounted for.
Incomplete feature windows around the gaps were discarded to prevent incorrect feature generation.

## 6. Known Limitations
- Adjacent feature frames overlap and are therefore correlated.
- The clips originate from a fixed provided audio pool.
- Boundary-crossing frames are excluded from the labelled dataset.
- The dataset should not be treated as a collection of independent real-world recordings.

## 7. Intended Use
This dataset is intended for the Edge AI coursework, feature extraction experiments
and keyword-classification model development. It should not be treated as a
representative dataset for unrestricted real-world speech recognition.
