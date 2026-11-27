import numpy as np
import matplotlib.pyplot as plt
from team_audio import get_clip

team = 11
clip_id = 120

# Load the silence clip at 48 kHz
x48 = get_clip(team, clip_id, fs=48000, antialias=False)

# Load at 8 kHz without anti-alias filtering
x8_raw = get_clip(team, clip_id, fs=8000, antialias=False)

# Load at 8 kHz with anti-alias filtering
x8_filtered = get_clip(team, clip_id, fs=8000, antialias=True)


def plot_spectrum(x, fs, title):
    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(len(x), d=1/fs)

    plt.plot(frequencies, spectrum)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plot_spectrum(x48, 48000, "Original Silence Clip - 48 kHz")

plt.subplot(3, 1, 2)
plot_spectrum(x8_raw, 8000, "8 kHz Without Anti-Alias Filtering")

plt.subplot(3, 1, 3)
plot_spectrum(x8_filtered, 8000, "8 kHz With Anti-Alias Filtering")

plt.tight_layout()
plt.savefig("results/mission2/graphs/aliasing_comparison.png")
plt.show()