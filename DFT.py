import numpy as np
import matplotlib.pyplot as plt

def compute_dft_symmetric(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    half_N = N // 2 + 1  # Compute only half due to symmetry
    for k in range(half_N):
        for n in range(N):
            twiddle = np.exp(-2j * np.pi * k * n / N)
            X[k] += x[n] * twiddle
        if k > 0 and k < N // 2:
            X[N - k] = np.conj(X[k])  # Use symmetry property
    return X

# Define the sequence x(n) = -(-1)^n for 0 <= n < 8
# Take input from user
input_str = input("Enter the sequence x(n) as space-separated values (e.g., 1 2 3 4): ")
x = np.array([float(val) for val in input_str.strip().split()])
N = len(x)

# Compute DFT using symmetry and periodicity properties
X = compute_dft_symmetric(x)

# Compute magnitude and phase
magnitude = np.abs(X)
phase = np.angle(X)

# Compute Twiddle factor matrix
W = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)

# Print Twiddle factor matrix
print("Twiddle Factor Matrix:")
print(W)

# Plot magnitude response
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.stem(np.arange(N), magnitude, basefmt=" ")
plt.xlabel("k")
plt.ylabel("|X(k)|")
plt.title("Magnitude Response")
plt.grid()

# Plot phase response
plt.subplot(1, 3, 2)
plt.stem(np.arange(N), phase, basefmt=" ")
plt.xlabel("k")
plt.ylabel("∠X(k) (radians)")
plt.title("Phase Response")
plt.grid()

# Plot Twiddle factor matrix
plt.subplot(1, 3, 3)
plt.imshow(np.angle(W), cmap='twilight', extent=[0, N-1, 0, N-1])
plt.colorbar(label='Phase (radians)')
plt.xlabel("n")
plt.ylabel("k")
plt.title("Twiddle Factor Matrix")

plt.tight_layout()
plt.show()
