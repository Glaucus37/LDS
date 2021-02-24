"""
@author: Bruno Hentschel (https://github.com/Glaucus37)
"""

# Section 0: Imports and declarations

import sys
import math
from cython import array
import numpy as np
cimport numpy as cnp
import random as rand
import matplotlib.pyplot as plt
from matplotlib import gridspec

# static variable definitions
cdef int D = 2 # Number of dimensions
cdef int N = 10 # Number of particles
cdef double t_max = 1e1 # maximum time to simulate
cdef double L = 10. # Dimension length
cdef double a_init = 2. # Initial acceleration
cdef double r_lim = 2.

cdef double dt = 1e-2 # Time step
cdef double dt_sq = dt ** 2
cdef double o_sqrt_dt = 1 / np.sqrt(dt) # over sqrt(dt)
cdef int max_steps = int(t_max / dt)

cdef int lat_size = 4
cdef double cell_size = L / lat_size
cdef long cells = lat_size ** 2
cdef int list_size = N + cells


# array declarations
cdef double [:, :] x = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] y = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] vx = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] vy = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] ax = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] ay = np.zeros((max_steps, N), dtype=np.double)
cdef double [:] kin_U = np.zeros(max_steps, dtype=np.double)
cdef long [:, :] k_neighbors = np.zeros((cells, 5), dtype=np.int32)
cdef int [:] k_list = np.zeros(list_size, dtype=np.int32)


# Section I: main function

# main routine which sets up and executes simulation with given parameters
# paramateres are initialized to 1 by default
# all parameters in computational units
cpdef void main((double, double, double, double) args=(1., 1., 1., 1.), int init=0):
  cdef double v_0

  a = accel(args) # initialize acceleration-class object
  v_0 = args[3] # initialize velocity from passed argument to main()
  setup(v_0, init) # set initial conditions (position, velocity, acceleration)
  run_sim(a) # execute simulation with a holding the appropiate params


# Section II: Function definition

# set all relevant arrays to zeros for the upcoming run
cdef void setup(double v_0, int init):
  global cells, max_steps, N
  rand.seed() # randomize the run's seed
  # Array declarations
  x = np.zeros((max_steps, N))
  y = np.zeros((max_steps, N))
  vx = np.zeros((max_steps, N))
  vy = np.zeros((max_steps, N))
  ax = np.zeros((max_steps, N))
  ay = np.zeros((max_steps, N))
  kin_U = np.zeros(max_steps)

  init_particles(v_0, init) # once arrays are initialized, set initial values
  set_neighbors()


# Initialize positions, velocities and accelerations for all particles
cdef void init_particles(double v_0, int init):
  cdef double theta
  cdef double vx_cm = 0.
  cdef double vy_cm = 0.

  cdef int i
  for i in range(N):

    if init == 0: # standard random initialization
      x[0, i] = L * <double>rand.random()
      y[0, i] = L * <double>rand.random()
    else: # set particles to specific position (current: center)
      x[0, i] = L * 0.5
      y[0, i] = L * 0.5

    # set a random angle for velocity and acceleration, with a given magnitude
    theta = <double>(2 * math.pi * rand.random())
    vx[0, i] = v_0 * math.cos(theta)
    vy[0, i] = v_0 * math.sin(theta)
    vx_cm += vx[0, i]
    vy_cm += vy[0, i]

    theta = <double>(2 * math.pi * rand.random())
    ax[0, i] = a_init * math.cos(theta)
    ay[0, i] = a_init * math.sin(theta)

  vx_cm /= N
  vy_cm /= N

  for i in range(N):
    vx[0, i] -= vx_cm
    vy[0, i] -= vy_cm


# Simulation loop
cdef void run_sim(object a):
  cdef double m = a.m # get mass of particles
  cdef int step = 0

  while step < max_steps - 1:
    next = step + 1
    # this is the main block of the simulation...
    verlet(step)
    a.run(step)
    vel_half_step(step)
    # ...more details later on how it's set up
    kin_U[step] = kin_energy(step, m) # keep track of energy

    step += 1

  # since all particles have an initial position, velocity, acceleration,
    # we need only (n - 1) iterations to obtain n sets of data
    # as such, we need to calculate the energy one last time:
  kin_U[step] = kin_energy(step, m)


# Verlet routine updates position, and does a half-step update on velocity
cdef void verlet(int step):
  cdef int next = step + 1
  cdef double x_new, y_new

  cdef int i
  for i in range(N):
    # x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
    # this equation is satisfied perfectly, however...
    # v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    # since we don't know a_{n+1} yet, we take only the first term,
      # and later update velocity a second time, after
      # acceleration has been calculated
    x_new = x[step, i] + vx[step, i] * dt + 0.5 * ax[step, i] * dt_sq
    y_new = y[step, i] + vy[step, i] * dt + 0.5 * ay[step, i] * dt_sq
    x[next, i] = pbc(x_new)
    y[next, i] = pbc(y_new)
    vx[next, i] = vx[step, i] + 0.5 * ax[step, i] * dt
    vy[next, i] = vy[step, i] + 0.5 * ay[step, i] * dt


# accel was initially a simple function, but updating global variables
  # (gamma, sigma, mass) had been a pain in the a**; defining this class
  # seems to have done the trick
cdef class accel:
  cdef public double gamma, sigma, m # variables that will be accessed outside
  def __init__(self, (double, double, double, double) args):
    # args have a default initialization at main()
    # the order of arguments accel recieves is critical! Beware any changes!
    self.gamma = args[0]
    self.sigma = o_sqrt_dt * np.sqrt(2 * args[0] * args[1] * args[2])
    self.m = args[2]
  def run(self, int step):
    # what was previously the accel() function became the run() method
    cdef double g1, g2, a_x, a_y
    cdef int next = step + 1

    BulkForce(next)

    cdef int i
    for i in range(N):
      g1, g2 = gauss() # pair of gaussian numbers
      # technically force, but normalized to ignore mass, the run()
      # method has two terms:
      # 1. -gamma * v -- dissipative/drag force
      # 2. sigma * g -- known as 'the derivative of a wiener process',
        # this term simmulates a thermodinamic system whose temperature is
        # described by a gaussian distribution of its particles' velocities
      ax[next, i] = -self.gamma * vx[step, i] + self.sigma * g1
      ay[next, i] = -self.gamma * vy[step, i] + self.sigma * g2


#Update velocity at half steps
cdef void vel_half_step(int step):
  cdef int next = step + 1

  cdef int i
  for i in range(N):
    # here is the missing term for the Verlet routine, since at this point
    # accel will have been called
    vx[next, i] += 0.5 * ax[next, i] * dt
    vy[next, i] += 0.5 * ay[next, i] * dt


# generate (and return) a pair of normally distributed numbers
cdef (double, double) gauss():
  cdef double fac, v1, v2
  cdef double r_sq = 0.
  cdef double s_dev = 1.
  while r_sq <= 0. or r_sq > 1.:
    v1 = 2. * rand.random() - 1.
    v2 = 2. * rand.random() - 1.
    r_sq = v1 ** 2 + v2 ** 2
  fac = s_dev * np.sqrt(-2. * np.log(r_sq) / r_sq)
  return (v1 * fac, v2 * fac)


# Periodic boundary conditions
cdef double pbc(double x):
  if x < 0:
    x += L
  elif x >= L:
    x -= L

  return x


# calculate rms velocity
cdef double rms():
  cdef double ms = 0

  cdef int i
  for i in range(N):
    ms += vx[-1, i] ** 2 + vy[-1, i] ** 2

  return np.sqrt(ms)


# calculate average kinetic energy of particles
cpdef double kin_energy(int step, double m):
  cdef double kin = 0
  cdef int i
  for i in range(N):
    kin += (vx[step, i] ** 2 + vy[step, i] ** 2)

  return 0.5 * kin / (N * m)


# Set neighbors for each cell
cdef void set_neighbors():
  # any cell always has 9 neighbors (including itself), 4 of
    # which will have it as their neighbor
  # as such, we can safely assign 5 neighbors to each cell
  global k_neighbors
  cdef long [:] naive_neighbors

  cdef int k, i
  for k in range(cells):
    # 'naive_neighbors' point towards the cells neighbors, if the boundaries
      # didn't exist
    naive_neighbors = np.array([0, 1, lat_size - 1, lat_size, lat_size + 1])
    for i in range(5):
      # 'move' neighbors to corresponding initial cell
      naive_neighbors[i] += k

    # Boundary conditions on neighbor list
    if k % lat_size == 0:
      naive_neighbors[2] += lat_size
    elif k % lat_size == lat_size - 1:
      naive_neighbors[1] -= lat_size
      naive_neighbors[4] -= lat_size
    if k // lat_size == lat_size - 1:
      naive_neighbors[2] -= cells
      naive_neighbors[3] -= cells
      naive_neighbors[4] -= cells

    k_neighbors[k] = naive_neighbors # append k'th neighbors to neighbors list


cdef void BulkForce(int step):
  global N, cells
  global x, y
  global k_list
  # k_list includes 'N' particles, followed by 'cells' cells
  cdef int i, j, n, k, m1, m2, j1, j2

  for j in range(N, list_size):
    k_list[j] = -1
    """
    try:
      k_list[j] = -1
    except IndexError:
      print 'j: ', j
    """

  for n in range(N):
    k = int(x[step, n] / cell_size) + int(y[step, n] / cell_size) * lat_size + N
    k_list[n] = k_list[k]
    k_list[k] = n
    """
    try:
      k = int(x[step, n] / cell_size) + int(y[step, n] / cell_size) * lat_size + N
      k_list[n] = k_list[k]
      k_list[k] = n
    except IndexError:
      print 'k, n: ', k, n
    """

  for k in range(cells):
    m1 = k + N
    for i in range(5):
      m2 = k_neighbors[k, i] + N
      j1 = k_list[m1]
      """
      try:
        m2 = k_neighbors[k, i] + N
        j1 = k_list[m1]
      except IndexError:
        print 'm2, j1: ', m2, j1
      """
      while j1 >= 0:
        j2 = k_list[m2]
        """
        try:
          j2 = k_list[m2]
        except IndexError:
          print 'j2: ', j2
        """
        while j2 >= 0:
          if j2 < j1 or m1 != m2:
            force(j1, m1, j2, m2, step)
          j2 = k_list[j2]
        j1 = k_list[j1]


cdef void force(j1, m1, j2, m2, step):
  global x, y
  global cell_size
  cdef double f_n
  cdef int next = step + 1

  cdef double dx, dy, rr
  dx = pbc(x[step, j2] - x[step, j1]) - L / 2
  dy = pbc(y[step, j2] - y[step, j1]) - L / 2
  rr = np.sqrt(dx ** 2 + dy ** 2)

  if rr < cell_size:
    f_n = LennerdJones(rr)
    fnx = f_n * dx / rr
    fny = f_n * dy / rr
    ax[next, j1] += fnx
    ay[next, j1] += fny
    ax[next, j2] -= fnx
    ay[next, j2] -= fny


cdef double LennerdJones(double r):
  return 24 / r * (2 * (1 / r)**12 - (1 / r)**6)


# Section III: Plots

# -s command, makes two plots:
  # 1. system's energy vs. t
  # 2. Boltzmann velocity distribution
cpdef void plot_simple():
  fig, axes = plt.subplots(2, 1, figsize=(8, 6))

  main()
  axes[0].plot(kin_U, linewidth=1)
  axes[0].plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  axes[0]. plot([0, max_steps], [0, 0], color='gray', linestyle='-')
  axes[0].set_xticks([0, max_steps / 2,  max_steps])
  axes[0].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[0].set_xlabel('Time (s)')
  axes[0].set_yticks([0, 1])
  axes[0].set_ylabel(r'Energy $(k_BT)$')
  axes[0].title.set_text('Energy over time')

  cdef double mu = 0.
  cdef double sig = 1.
  count, bins, ignored = axes[1].hist(vx, 80, density=True)
  axes[1].plot(bins, 1/(sig * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mu)**2 / (2 * sig**2) ),
            linewidth=2, color='r', linestyle='-.',
            label=r'$\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}$')
  axes[1].set_xlim([np.percentile(bins, 0.5), np.percentile(bins, 99.5)])
  axes[1].set_xlabel('Velocity')
  axes[1].set_ylabel('Relative Frequency')
  axes[1].title.set_text('Histogram of Particle Velocity')
  axes[1].legend()

  fig.tight_layout(pad=3.)
  plt.show()


# empty command, makes 4 plots, all showing the effects of varying a
  # single parameter.
    # 1. gamma
    # 2. kBT
    # 3. mass
    # 4. initial velocity
cpdef void plot_full():
  fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=True)
  fig.suptitle('Dependency of Energy on Initial Conditions')

  axes[0, 0].plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  axes[0, 0].plot([0, max_steps], [0, 0], color='gray', linestyle='-')
  axes[0, 1].plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  axes[0, 1].plot([0, max_steps], [2, 2], color='r', linestyle='-.')
  axes[0, 1].plot([0, max_steps], [0.5, 0.5], color='r', linestyle='-.')
  axes[0, 1].plot([0, max_steps], [0, 0], color='gray', linestyle='-')
  axes[1, 0].plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  axes[1, 0].plot([0, max_steps], [0, 0], color='gray', linestyle='-')
  axes[1, 1].plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  axes[1, 1].plot([0, max_steps], [0, 0], color='gray', linestyle='-')

  main()
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 1.0', linewidth=1)
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 1.0', linewidth=1)
  axes[1, 0].plot(kin_U, label='m = 1.0', linewidth=1)
  axes[1, 1].plot(kin_U, label=r'$v_0 = 1.0$', linewidth=1)

  main((2., 1., 1., 1.))
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 2.0', linewidth=1)
  main((0.5, 1., 1., 1.))
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 0.5', linewidth=1)
  axes[0, 0].set_xlim(-0.1, max_steps / 8)
  axes[0, 0].set_xticks([0, max_steps / 16,  max_steps / 8])
  axes[0, 0].set_xticklabels([0, max_steps * dt / 16, max_steps * dt / 8])
  axes[0, 0].set_ylabel(r'Energy $(k_BT)$')
  axes[0, 0].legend()

  main((1., 2., 1., 1.))
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 2.0', linewidth=1)
  main((1., 0.5, 1., 1.))
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 0.5', linewidth=1)
  axes[0, 1].set_xticks([0, max_steps / 2,  max_steps])
  axes[0, 1].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[0, 1].legend()

  main((1., 1., 2., 1.))
  axes[1, 0].plot(kin_U, label='m = 2.0', linewidth=1)
  main((1., 1., 0.5, 1.))
  axes[1, 0].plot(kin_U, label='m = 0.5', linewidth=1)
  axes[1, 0].set_xticks([0, max_steps / 2,  max_steps])
  axes[1, 0].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[1, 0].set_ylabel(r'Energy $(k_BT)$')
  axes[1, 0].set_xlabel('Time (s)')
  axes[1, 0].legend()

  main((1., 1., 1., 2.))
  axes[1, 1].plot(kin_U, label=r'$v_0 = 2.0$', linewidth=1)
  main((1., 1., 1., 0.5))
  axes[1, 1].plot(kin_U, label=r'$v_0 = 0.5$', linewidth=1)
  axes[1 ,1].set_xticks([0, max_steps / 2,  max_steps])
  axes[1 ,1].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[1, 1].set_xlabel('Time (s)')
  axes[1, 1].legend()

  axes[0, 0].set_yticks([0, 1, 2])

  plt.tight_layout(pad=1.)

  plt.show()


# -p command, makes two scatter-plots:
  # 1. initial positions of particles
  # 2. end positions of particles
cpdef void plot_pos():
  global k_list

  fig, axes = plt.subplots(figsize=(8, 6))
  fig.suptitle('Distribution of particles')

  main((1., 1., 1., 1.), 1)

  cdef int k
  for k in range(list_size):
    print k, k_list[k]

  plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
  plt.scatter(x[0], y[0])
  plt.xlim(0, 10)
  plt.xticks([0, 10])
  plt.xlabel('x')
  plt.ylim(0, 10)
  plt.yticks([0, 10])
  plt.ylabel('y')
  plt.title('t = 0s')

  plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
  cdef int i, j
  for i in range(lat_size):
    if i:
      for j in range(lat_size):
        if j:
          plt.plot([0, L], [j * cell_size, j * cell_size], color='gray', linestyle='-')
          plt.plot([i * cell_size, i * cell_size], [0, L], color='gray', linestyle='-')
  plt.scatter(x[-1], y[-1])
  plt.xlim(0, 10)
  plt.xticks([0, 10])
  plt.xlabel('x')
  plt.ylim(0, 10)
  plt.yticks([0, 10])
  plt.ylabel('y')
  plt.title('t = {}s'.format(t_max))

  plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
  plt.plot([0, max_steps], [0, 0], color='gray', linestyle='-')
  plt.plot([0, max_steps], [1, 1], color='r', linestyle='-.')
  plt.plot(kin_U, linewidth=1)
  plt.xticks([0, max_steps / 2, max_steps], [0, t_max / 2, t_max])
  plt.xlabel('Time (s)')
  plt.ylabel(r'Energy $(k_BT)$')

  plt.tight_layout(pad=2.)

  plt.show()
