import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand



if __name__ == '__main__':
    arg = sys.argv[1:]

    if '-q' not in arg:
        if '-s' in arg:
            f.plot_simple()
        elif '-p' in arg:
            f.plot_pos()
        else:
            f.plot_full()
