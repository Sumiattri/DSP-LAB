import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def overlap_save(x, h, N):
    M = len(h)
    L = N - M + 1
    H = np.fft.fft(h, N)
    x_padded = np.concatenate((np.zeros(M - 1), x))
    y = []

    for i in range(0, len(x_padded) - N + 1, L):
        block = x_padded[i:i+N]
        X = np.fft.fft(block)
        Y = X * H
        y_block = np.fft.ifft(Y).real
        y.extend(y_block[M-1:])

    return np.array(y)

# ------------------ User Input ------------------ #

# Input sequence
x_input = input("Enter input sequence x[n] (space-separated numbers): ")
x = np.array([float(val) for val in x_input.strip().split()])

# Filter sequence
h_input = input("Enter filter h[n] (impulse response, space-separated): ")
h = np.array([float(val) for val in h_input.strip().split()])

# FFT block size
N = int(input(f"Enter FFT block size N (>= {len(h)}): "))
if N < len(h):
    raise ValueError("FFT block size N must be >= length of filter h[n]")

# ------------------ Filtering ------------------ #

y_overlap = overlap_save(x, h, N)
y_direct = lfilter(h, [1], x)

# ------------------ Plotting ------------------ #

plt.figure(figsize=(12, 5))
plt.plot(y_direct, label="Direct Convolution", alpha=0.6)
plt.plot(y_overlap, '--', label="Overlap-Save Output", alpha=0.8)
plt.title("Filtering using Overlap-Save Method")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()