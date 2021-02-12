LMS - Langevin Mechanics Simulation. 

Program structure:
setup.py - behind the scenes setup for interaction between python and C.
functions.pyx - Cython code for simulation.
main.py - read command line, import setup, call functions.pyx

Command line prompts:
      - no prompt: run full program, comparing variation of parameters.
-q    - quiet: run program without graphs
-s    - simple: plot only one instance, as well as its velocity distribution

Deprecated:
main.c
plots.py
plot.py
