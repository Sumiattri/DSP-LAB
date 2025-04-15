import matplotlib.pyplot as plt
import numpy as np

# Input sequences
x = np.array([1, 2, 3])
h = np.array([0, 1, 0.5])

# Lengths of the input signals
N = len(x)
M = len(h)

# Output length of the convolution result
L = N + M - 1
y = [0] * L  # Initialize output with zeros

# Perform linear convolution
for n in range(L):
    for k in range(N):
        if 0 <= n - k < M:
            y[n] += x[k] * h[n - k]

# Print the result
print("Linear convolution result:", y)

# Plot the result
plt.stem(range(len(y)), y)
plt.title("Linear Convolution Output")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
