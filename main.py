import sys

import functions as f


if __name__ == '__main__':
    arg = sys.argv[1:]

    # handle cases based on command line prompts
    if '-q' not in arg:
        if '-s' in arg:
            f.plot_simple()
        elif '-p' in arg:
            f.plot_pos()
        else:
            f.plot_full()
    else:
        f.main()
