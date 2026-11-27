# Mission 5 — Laptop vs Board Feature Comparison and Classification

## 1. Aim

The objective of Mission 5 was to evaluate the accuracy of the ESP32-S3 feature extraction pipeline against laptop-generated reference features. The experiment also examined the effect of quantisation, sampling rate and anti-alias filtering on feature accuracy and classification performance.

## 2. Experimental Setup

- Team: 11
- Board: ESP32-S3-DevKitC-1
- Audio classes: on, off, stop, go, _unknown and _silence
- Total clips: 144 (24 per class)
- Feature dimensions: 40 Mel bands
- Window length: 480 samples
- Hop length: 320 samples
- FFT size: 512
- Total labelled feature frames: 7,056

## 3. Baseline Feature Comparison

The original float32 audio clips were processed on the laptop using the reference DSP implementation. The resulting features were compared frame-by-frame against the board-generated features.

| Metric | Result |
|---|---:|
| Frames compared | 7,056 |
| Mean absolute error | 0.005542 dB |
| Worst absolute error | 0.107699 dB |

The results demonstrate close agreement between the laptop reference and the board implementation.

### Separating Quantisation and DSP Error

To isolate numerical differences in feature extraction, the board output was also compared against NumPy features generated from the same quantised int16 input.

| Metric | Result |
|---|---:|
| Mean absolute error | 0.000265 dB |
| Worst absolute error | 0.000508 dB |

This indicates that the board's feature extraction implementation closely matches the laptop reference when both process identical input samples. The larger difference in the original comparison is primarily associated with the different input representations, particularly quantisation.

## 4. Alternative Configuration Experiments

Three alternative configurations were evaluated using the recorded board features and corrected laptop references.

| Configuration | Mean absolute error (dB) | Worst error (dB) |
|---|---:|---:|
| int8, S=1 | 1.3051 | 15.5124 |
| int8, S=0.25 | 1.0587 | 34.4191 |
| int16, 8 kHz, no filter | 8.0697 | 49.7558 |

The int8 configurations produced larger errors than the int16 baseline. Although S=0.25 had a lower mean error than S=1, its worst-case error was considerably higher.

The 8 kHz unfiltered configuration produced the largest feature mismatch against the filtered laptop reference. This demonstrates the importance of anti-alias filtering when reducing the sampling rate.

The corresponding error heatmaps are stored in the Mission 5 graphs directory.

## 5. Classification Experiment

A nearest-centroid classifier was implemented using per-clip mean and standard deviation values for each of the 40 Mel bands.

Each clip was represented by 80 features:

- 40 Mel-band mean values.
- 40 Mel-band standard deviation values.

The dataset was divided into 16 training clips and 8 testing clips per class. This resulted in 96 training clips and 48 testing clips. The same split was used for every configuration.

Feature standardisation was fitted using training data only. The classifier was trained using laptop features and evaluated using both laptop and board features.

### Classification Results

| Configuration | Laptop accuracy | Board accuracy |
|---|---:|---:|
| int16, 16 kHz, filtered | 81.25% | 81.25% |
| int8, S=1 | 81.25% | 66.67% |
| int8, S=0.25 | 81.25% | 50.00% |
| int16, 8 kHz, no filter | 60.42% | 16.67% |

The int16 filtered baseline maintained identical laptop and board classification accuracy. The int8 configurations showed reduced board accuracy, while the 8 kHz unfiltered configuration performed poorly.

## 6. Final Configuration Selection

The selected configuration is:

**16 kHz sampling rate, anti-alias filtering enabled, int16 audio format and scale 1.0.**

The decision is supported by the following observations:

1. The baseline achieved 81.25% classification accuracy on both laptop and board features.
2. The mean feature error against the original float32 laptop reference was only 0.005542 dB.
3. The same-quantised-input comparison showed a mean numerical error of only 0.000265 dB.
4. Both int8 alternatives caused measurable classification degradation.
5. The 8 kHz unfiltered configuration produced substantially worse feature and classification results.

The baseline is therefore the most reliable configuration among the tested alternatives.

## 7. Limitations

- Classification evaluation used 48 held-out clips, so the measured accuracy should not be interpreted as production-level performance.
- The nearest-centroid classifier is a simple baseline rather than a fully optimised recognition model.
- The results are specific to Team 11's generated audio dataset and experimental conditions.
- The 8 kHz experiment changes both sampling rate and board filtering, so their individual effects cannot be completely separated from this comparison alone.
- Worst-case feature errors indicate that average error alone is insufficient to describe the accuracy of a configuration.

## 8. Conclusion

Mission 5 demonstrated that the ESP32-S3 can reproduce the laptop reference feature extraction closely when operating with 16 kHz filtered int16 audio.

The classification experiment further showed that the selected baseline configuration preserves the measured laptop classification accuracy on the board. The results support using 16 kHz filtered int16 audio as the deployment configuration for the doorbell system.