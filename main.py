import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand


# if __name__ == '__main__':

    fig, ax = plt.subplots(nrows=2, ncols=1)

    try:
        arr = np.array(sys.argv[1:])
        v_arr = arr.astype(float)
        for i in len(v_arr):
            v_arr[i] = float(v_arr[i])
    except IndexError:
        v_arr = [1.]
    except TypeError:
        'input error'
    """
    for v in v_arr:
        kin_U, v_x, v_max = f.main(v)
        v_max += 10

        t = range(len(kin_U))
        ax[0].plot(t, kin_U)
    """
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)

    x = np.linspace(-0.02, 0.02, 100)
    count, bins, ignores = plt.hist(v_x[100:, :], 30, range=(-.01, .01))
    # ax[0].plot([0, t[-1]], [1, 1])
    ax[1].plot(x, norm.pdf(x), alpha=0.6)
    # ax[1].xticks((0, ))
    # plt.ylim((0, 1))
    # plt.xlim((-.02, .02))
    plt.show()
