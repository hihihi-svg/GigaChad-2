import numpy as np
from team_audio import get_clip
from experiments.mission2.quantisation import quantise, pack, unpack

word = get_clip(11, 0, fs=16000, antialias=True)
silence = get_clip(11, 120, fs=16000, antialias=True)

clips = [
    ("Word", word),
    ("Silence", silence)
]

configs = [
    (16, 1.0),
    (8, 1.0),
    (8, 0.25)
]

for clip_name, x in clips:
    for bits, S in configs:

        # 1. Quantise
        q = quantise(x, bits, S)

        # 2. Pack
        data = pack(q, bits)

        # 3. Unpack
        q_unpacked = unpack(data, bits)

        # 4. Reconstruct the original audio
        levels = 2 ** (bits - 1) - 1
        x_reconstructed = q_unpacked * S / levels

        # 5. Calculate original signal power
        signal_power = np.mean(x ** 2)

        # 6. Calculate quantisation error power
        error = x - x_reconstructed
        error_power = np.mean(error ** 2)

        # 7. Calculate SQNR
        sqnr = 10 * np.log10(signal_power / error_power)

        # 8. Count clipped samples
        clipped = np.sum(np.abs(x) > S)

        # 9. Print results
        print(
            f"{clip_name} - int{bits}, S={S}: "
            f"SQNR = {sqnr:.2f} dB, "
            f"Clipped Samples = {clipped}"
        )