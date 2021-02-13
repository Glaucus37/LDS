import sys
import math
from cython import array
import numpy as np
cimport numpy as cnp
import random as rand
import matplotlib.pyplot as plt
from matplotlib import gridspec

# static variable definitions
cdef int D = 2
cdef int N = 100
cdef double t_max = 1e1
cdef double L = 10.
cdef double a_init = 2.

cdef double dt = 1e-2
cdef double dt_sq = dt ** 2
cdef double o_sqrt_dt = 1 / math.sqrt(dt)
cdef int max_steps = int(t_max / dt)

cdef int lat_size = 5
cdef long cells = lat_size ** 2


# array declarations
cdef double [:, :] x = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] y = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] vx = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] vy = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] ax = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] ay = np.zeros((max_steps, N), dtype=np.double)
cdef double [:] kin_U = np.zeros(max_steps, dtype=np.double)
# cdef long [:, :] k_neighbors = np.zeros((cells, 5))



# main routine which sets up and executes simulation with given parameters
# paramateres are initialized to 1 by default
# all parameters in computational units
cpdef void main(object args=(1., 1., 1., 1.), int init=0):
  cdef double v_0

  a = accel(args) # initialize acceleration-class object
  v_0 = args[3] # initialize velocity from passed argument to main()
  setup(v_0, init) # set initial conditions (position, velocity, acceleration)
  run_sim(a) # execute simulation with a holding the appropiate params


# set all relevant arrays to zeros for the upcoming run
cdef void setup(double v_0, int init):
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


# Initialize positions, velocities and accelerations for all particles
cdef void init_particles(double v_0, int init):
  cdef double theta

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

    theta = <double>(2 * math.pi * rand.random())
    ax[0, i] = a_init * math.cos(theta)
    ay[0, i] = a_init * math.sin(theta)


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
  def __init__(self, args):
    # args have a default initialization at main()
    # the order of arguments accel recieves is critical! Beware any changes!
    self.gamma = args[0]
    self.sigma = o_sqrt_dt * math.sqrt(2 * args[0] * args[1] * args[2])
    self.m = args[2]
  def run(self, int step):
    # what was previously the accel() function became the run() method
    cdef double g1, g2, a_x, a_y
    cdef int next = step + 1

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
  fac = s_dev * math.sqrt(-2. * math.log(r_sq) / r_sq)
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
cdef long [:, :] set_neighbors():
  # any cell always has 9 neighbors (including itself), 4 of
    # which will have it as their neighbor
  # as such, we can safely assign 5 neighbors to each cell
  cdef long [:, :] neighbors = np.zeros((cells, 5), dtype=np.int32)
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

    neighbors[k] = naive_neighbors # append k'th neighbors to neighbors list

  return neighbors


# -s command, makes two plots:
  # 1. system's energy vs. t
  # 2. Boltzmann velocity distribution
cpdef void plot_simple():
  fig, axes = plt.subplots(2, 1, figsize=(8, 6))

  main()
  axes[0].plot(kin_U)
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
            linewidth=2, color='gray', linestyle='--',
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
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 1.0')
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 1.0')
  axes[1, 0].plot(kin_U, label='m = 1.0')
  axes[1, 1].plot(kin_U, label=r'$v_0 = 1.0$')

  main((2., 1., 1., 1.))
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 2.0')
  main((0.5, 1., 1., 1.))
  axes[0, 0].plot(kin_U, label=r'$\gamma$ = 0.5')
  axes[0, 0].set_xlim(-0.1, max_steps / 8)
  axes[0, 0].set_xticks([0, max_steps / 16,  max_steps / 8])
  axes[0, 0].set_xticklabels([0, max_steps * dt / 16, max_steps * dt / 8])
  axes[0, 0].set_ylabel(r'Energy $(k_BT)$')
  axes[0, 0].legend()

  main((1., 2., 1., 1.))
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 2.0')
  main((1., 0.5, 1., 1.))
  axes[0, 1].plot(kin_U, label=r'$k_BT$ = 0.5')
  axes[0, 1].set_xticks([0, max_steps / 2,  max_steps])
  axes[0, 1].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[0, 1].legend()

  main((1., 1., 2., 1.))
  axes[1, 0].plot(kin_U, label='m = 2.0')
  main((1., 1., 0.5, 1.))
  axes[1, 0].plot(kin_U, label='m = 0.5')
  axes[1, 0].set_xticks([0, max_steps / 2,  max_steps])
  axes[1, 0].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[1, 0].set_ylabel(r'Energy $(k_BT)$')
  axes[1, 0].set_xlabel('Time (s)')
  axes[1, 0].legend()

  main((1., 1., 1., 2.))
  axes[1, 1].plot(kin_U, label=r'$v_0 = 2.0$')
  main((1., 1., 1., 0.5))
  axes[1, 1].plot(kin_U, label=r'$v_0 = 0.5$')
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
  fig, axes = plt.subplots(figsize=(8, 6))
  fig.suptitle('Distribution of particles')

  main((1., 1., 1., 1.), 1)

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
  plt.plot(kin_U)
  plt.xticks([0, max_steps / 2, max_steps], [0, t_max / 2, t_max])
  plt.xlabel('Time (s)')
  plt.ylabel(r'Energy $(k_BT)$')

  plt.tight_layout(pad=2.)

  plt.show()
