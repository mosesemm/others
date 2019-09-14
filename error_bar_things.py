import numpy as np
import matplotlib.pyplot as plt

x = [-97.5, -107.2, -141.06] #np.arange(0.1, 4, 0.5)
y = [0, 0.3, 0.5] #np.exp(-x)

# example variable error bar values
#yerr = 0.1 + 0.2*np.sqrt(x)
yerr = y - np.mean(y)
xerr = 0.1 + yerr

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='o')
plt.show()

