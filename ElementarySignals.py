import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)  # Time index

# Impulse
impulse = np.where(n == 0, 1, 0)

# Step
step = np.where(n >= 0, 1, 0)

# Ramp
ramp = np.where(n >= 0, n, 0)

# Exponential
a = 0.8
exponential = a ** n

# Sinusoidal
A = 1
w = 0.2 * np.pi
phi = 0
sinusoidal = A * np.sin(w * n + phi)

# Plotting
plt.figure(figsize=(12, 8))

signals = [impulse, step, ramp, exponential, sinusoidal]
titles = ['Impulse', 'Step', 'Ramp', 'Exponential', 'Sinusoidal']

for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.stem(n, signals[i])
    plt.title(titles[i])
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()