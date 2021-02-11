import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand


if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=2, ncols=1)

    v_arr = []
    for x in sys.argv:
        if not x == 'main.py':
            v_arr.append(float(x))
    if v_arr == []:
        v_arr = [1.]
    print(v_arr)

    for v in v_arr:
        kin_U, v_x, max_steps = f.main(v)
        ax[0].plot(kin_U)
        for i in range(len(kin_U)):
            print(kin_U[i])

    ax[0].plot([0, max_steps], [1, 1], 'k-')

    count, bins, ignored = plt.hist(v_x, 80, density=True)
    ax[1].plot(bins, 1/(1 * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - 0)**2 / (2 * 1**2) ),
            linewidth=2, color='k')
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    x = np.arange(-1, 1, 0.1)
    # ax[1].xticks((0, ))
    # plt.ylim((0, 1))
    # plt.xlim((-.02, .02))
    plt.show()
