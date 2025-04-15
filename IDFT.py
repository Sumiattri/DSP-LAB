import numpy as np
import matplotlib.pyplot as plt

def compute_idft_symmetric(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    half_N = N // 2 + 1  # compute using half due to symmetry

    for n in range(N):
        for k in range(half_N):
            twiddle = np.exp(2j * np.pi * k * n / N)
            x[n] += X[k] * twiddle
            if k > 0 and k < N // 2:
                x[n] += np.conj(X[k]) * np.exp(2j * np.pi * (N - k) * n / N)
        x[n] /= N
    return x

# Take input for X(k) from user
input_str = input("Enter the sequence X(k) as space-separated real or complex values (e.g., 4 0 0 0 or 1+1j 0 0 0): ")
X = np.array([complex(val.replace('i', 'j')) for val in input_str.strip().split()])
N = len(X)

# Compute IDFT manually
x = compute_idft_symmetric(X)

# Twiddle Factor Matrix for IDFT
W_inv = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)

# Print Twiddle factor matrix
print("\nTwiddle Factor Matrix (for IDFT):")
print(np.round(W_inv, 3))  # Rounded for neatness

# Plot real and imaginary parts of the result
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(np.arange(N), x.real, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Re{x(n)}")
plt.title("Real Part of IDFT Result")
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(np.arange(N), x.imag, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Im{x(n)}")
plt.title("Imaginary Part of IDFT Result")
plt.grid()

plt.tight_layout()
plt.show()