import numpy as np

def quantise(x, bits, S):
    levels = 2 ** (bits - 1) - 1

    q = np.round(x / S * levels)

    q = np.clip(q, -(2 ** (bits - 1)), levels)

    return q

def pack(q, bits):
    if bits == 16:
        return q.astype("<i2").tobytes()

    elif bits == 8:
        return q.astype(np.int8).tobytes()

    else:
        raise ValueError("Only 8-bit and 16-bit quantisation are supported")

def unpack(data, bits):
    if bits == 16:
        return np.frombuffer(data, dtype="<i2")

    elif bits == 8:
        return np.frombuffer(data, dtype=np.int8)

    else:
        raise ValueError("Only 8-bit and 16-bit formats are supported")