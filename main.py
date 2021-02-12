import sys
import math

import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# import random as rand



if __name__ == '__main__':
    arg = sys.argv[1:]

    f.main()

    if '-q' not in arg:
        c = False
        if '-c' in arg:
            c = True
        f.plots(c)
