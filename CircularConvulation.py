import numpy as np
import matplotlib.pyplot as plt

# Input sequences (make sure both are same length N)
x = [1, 2, 3, 4]

h = [1, 0, 1, 0]
N = len(x)

# Zero padding if needed (optional if lengths already match)
if len(h) != N:
    if len(h) < N:
        h += [0] * (N - len(h))
    else:
        x += [0] * (len(h) - N)
        N = len(h)

print("x:", x)
print("h:", h)
# Output array
y = [0] * N

# Circular convolution logic
for n in range(N):
    for k in range(N):
        y[n] += x[k] * h[(n - k) % N]

print(y)

# Plotting
plt.stem(range(N), y)
plt.title("Circular Convolution")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

