import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand


if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=2, ncols=1)


    kin_U, v_x, max_steps = f.main()
    ax[0].plot(kin_U)
    ax[0].plot([0, max_steps], [1, 1], 'k-')

    count, bins, ignored = ax[1].hist(v_x, 80, density=True)
    ax[1].plot(bins, 1/(1 * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - 0)**2 / (2 * 1**2) ),
            linewidth=2, color='k')
    x = np.arange(-1, 1, 0.1)
    # ax[1].xticks((0, ))
    # plt.ylim((0, 1))
    # plt.xlim((-.02, .02))
    plt.show()
