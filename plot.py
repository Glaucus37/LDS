import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

f = open('velocity.txt', 'r')

vel = []

for line in f:
    vel.append(float(line))

fig, ax = plt.subplots(2, 1)

ax[0].plot(vel)
plt.hist(vel[150000:], bins=30, density=True)
x = np.arange(-4, 4, 0.1)
print(x)
ax[1].plot(norm.pdf(x, 0, 1))

plt.show()
