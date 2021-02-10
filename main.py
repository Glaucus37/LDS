import sys
import math

import functions as f



if __name__ == '__main__':
    # v_init = 5.

    try:
        v_init = float(sys.argv[1])
    except IndexError:
        v_init = 5.
    except TypeError:
        'speed must be floating point'
    v_0 = f.main(v_init)

    for i in range(len(v_0)):
        for j in range(len(v_0[0])):
            print(v_0[i, j])
        print('')

    try:
        v_init = float(sys.argv[2])
        f.main(v_init)
    except IndexError:
        pass
    except TypeError:
        'speed must be floating point'
