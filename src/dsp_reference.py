
import numpy as np


def features(x, fs=16000, win=480, hop=320, nfft=512, mels=40, mfcc=0):
    """
    Extract log-mel features or MFCCs from an audio signal.

    Parameters:
        x     : 1D audio signal
        fs    : Sampling frequency
        win   : Frame length in samples
        hop   : Hop length in samples
        nfft  : FFT size (power of 2, >= win)
        mels  : Number of mel filters
        mfcc  : Number of MFCC coefficients (0 = return log-mel)

    Returns:
        2D NumPy array:
        (number_of_frames, mels or mfcc)
    """

    x = np.asarray(x, dtype=np.float32)

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    if x.ndim != 1:
        raise ValueError(f"Input signal must be 1D, got {x.shape}")

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if win <= 0 or hop <= 0:
        raise ValueError("win and hop must be positive")

    if mels <= 0:
        raise ValueError("Number of mel filters must be positive")

    if nfft < win:
        raise ValueError("nfft must be >= win")

    if nfft & (nfft - 1):
        raise ValueError("nfft must be a power of 2")

    if mfcc < 0 or mfcc > mels:
        raise ValueError("mfcc must be between 0 and mels")

    if len(x) < win:
        raise ValueError(
            f"Signal length ({len(x)}) must be >= frame length ({win})"
        )

    # --------------------------------------------------
    # TODO 1: Divide signal into overlapping frames
    # --------------------------------------------------

    number_of_frames = 1 + (len(x) - win) // hop

    # --------------------------------------------------
    # TODO 2: Apply Hann window to each frame
    # --------------------------------------------------

    window = np.hanning(win)

    frames = np.array([
        x[i * hop : i * hop + win] * window
        for i in range(number_of_frames)
    ])

    # --------------------------------------------------
    # TODO 3: Zero-pad frames and calculate FFT
    # --------------------------------------------------

    zero_padded_frames = np.pad(
        frames,
        ((0, 0), (0, nfft - win)),
        mode="constant"
    )

    fft_frames = np.fft.rfft(
        zero_padded_frames,
        n=nfft
    )

    # --------------------------------------------------
    # TODO 4: Calculate power spectrum
    # --------------------------------------------------

    power_spectrum = np.abs(fft_frames) ** 2 / nfft

    # --------------------------------------------------
    # TODO 5: Create triangular mel filters
    # --------------------------------------------------

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Generate equally spaced mel boundaries from 20 Hz
    # to the Nyquist frequency.

    mel_points = np.linspace(
        hz_to_mel(20),
        hz_to_mel(fs / 2),
        mels + 2
    )

    mel_freqs = mel_to_hz(mel_points)

    # Actual frequencies represented by the FFT bins.

    fft_freqs = np.arange(nfft // 2 + 1) * fs / nfft

    mel_filters = np.zeros(
        (mels, nfft // 2 + 1),
        dtype=np.float64
    )

    # Construct triangular filters.

    for i in range(mels):

        left = mel_freqs[i]
        centre = mel_freqs[i + 1]
        right = mel_freqs[i + 2]

        # Rising slope
        rising = (fft_freqs - left) / (centre - left)

        # Falling slope
        falling = (right - fft_freqs) / (right - centre)

        # Combine both slopes into a triangular filter.
        mel_filters[i] = np.maximum(
            0,
            np.minimum(rising, falling)
        )

    # Check that every filter contains FFT-bin contributions.

    if np.any(np.sum(mel_filters, axis=1) == 0):
        raise ValueError(
            "One or more mel filters contain no FFT bins"
        )

    # --------------------------------------------------
    # TODO 6: Apply mel filters and convert to dB
    # --------------------------------------------------

    mel_energies = np.dot(
        power_spectrum,
        mel_filters.T
    )

    # Apply numerical floor BEFORE logarithm.

    mel_energies_db = 10 * np.log10(
        np.maximum(mel_energies, 1e-10)
    )

    # Do not clamp negative dB values.
    # Negative dB values are valid.

    # --------------------------------------------------
    # TODO 7: Apply orthonormal DCT-II for MFCCs
    # --------------------------------------------------

    if mfcc > 0:

        dct_matrix = np.zeros((mfcc, mels))

        for k in range(mfcc):
            for n in range(mels):

                dct_matrix[k, n] = np.cos(
                    np.pi * k * (2 * n + 1) / (2 * mels)
                )

        # Orthonormal DCT-II normalization.

        dct_matrix *= np.sqrt(2 / mels)

        dct_matrix[0, :] *= 1 / np.sqrt(2)

        mfccs = np.dot(
            mel_energies_db,
            dct_matrix.T
        )

        return mfccs

    return mel_energies_db
