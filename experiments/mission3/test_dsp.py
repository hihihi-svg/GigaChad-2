import numpy as np
from src.dsp_reference import features

# Generate a 1-second test signal
fs = 16000
x = np.random.randn(fs).astype(np.float32)

# Test log-mel output
mel = features(x, fs=fs, win=480, hop=320,
               nfft=512, mels=40, mfcc=0)

print("Log-mel shape:", mel.shape)
print("Contains NaN:", np.isnan(mel).any())
print("Contains Inf:", np.isinf(mel).any())

# Test MFCC output
mfcc = features(x, fs=fs, win=480, hop=320,
                nfft=512, mels=40, mfcc=13)

print("MFCC shape:", mfcc.shape)