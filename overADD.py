import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def overlap_add(x, h, N):
    M = len(h)
    L = N - M + 1
    H = np.fft.fft(h, N)
    y = np.zeros(len(x) + M - 1)  # Full convolution size

    for i in range(0, len(x), L):
        x_block = x[i:i+L]
        if len(x_block) < L:
            x_block = np.pad(x_block, (0, L - len(x_block)))  # Pad short blocks

        x_block_padded = np.pad(x_block, (0, M - 1))  # Pad to length N
        X = np.fft.fft(x_block_padded)
        Y = X * H
        y_block = np.fft.ifft(Y).real

        y[i:i+N] += y_block  # Add overlapped blocks
    return y

# ------------------ User Input ------------------ #

# Input sequence
x_input = input("Enter input sequence x[n] (space-separated): ")
x = np.array([float(val) for val in x_input.strip().split()])

# Filter sequence
h_input = input("Enter filter h[n] (impulse response, space-separated): ")
h = np.array([float(val) for val in h_input.strip().split()])

# FFT block size
N = int(input(f"Enter FFT block size N (>= {len(h)}): "))
if N < len(h):
    raise ValueError("FFT block size N must be >= length of filter h[n]")

# ------------------ Filtering ------------------ #

y_oa = overlap_add(x, h, N)
y_direct = lfilter(h, [1], x)

# ------------------ Plotting ------------------ #

plt.figure(figsize=(12, 5))
plt.plot(y_direct, label="Direct Convolution", alpha=0.6)
plt.plot(y_oa, '--', label="Overlap-Add Output", alpha=0.8)
plt.title("Filtering using Overlap-Add Method")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
