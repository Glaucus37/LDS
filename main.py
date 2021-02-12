import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand



if __name__ == '__main__':
    arg = sys.argv[1:]
    const = [1., 1., 1.]

    f.main()

    if '-q' not in arg:
        if '-s' in arg:
            f.plot_simple()
        else:
            f.plot_full()
