import sys
import math
from cython import array
import numpy as np
cimport numpy as cnp
import random as rand
import matplotlib.pyplot as plt

# from libc.stdlib cimport rand, srand, RAND_MAX

cython: language_level=3

# Variable definitions
cdef int D = 2
cdef int N = 100
cdef double dt = 1e-2
cdef double t_max = 1e1
cdef double L = 10.
cdef double v_init = 1.
cdef double a_init = 2.

cdef double dt_sq = dt ** 2
cdef double o_sqrt_dt = 1 / math.sqrt(dt)
cdef int max_steps = int(t_max / dt)

cdef int lat_size = 5
cdef long cells = lat_size ** 2


# Array declarations
cdef double [:, :] x = np.zeros((max_steps, N), dtype=np.double)
cdef double [:, :] y = np.zeros((max_steps, N))
cdef double [:, :] vx = np.zeros((max_steps, N))
cdef double [:, :] vy = np.zeros((max_steps, N))
cdef double [:, :] ax = np.zeros((max_steps, N))
cdef double [:, :] ay = np.zeros((max_steps, N))
cdef double [:] kin_U = np.zeros(max_steps)
# cdef long [:, :] k_neighbors = np.zeros((cells, 5))


cpdef void main(object args=(1., 1., 1.)):
  a = accel(args)
  setup()
  run_sim(a)

  return


cdef void setup():
  rand.seed()
  # Array declarations
  x = np.zeros((max_steps, N))
  y = np.zeros((max_steps, N))
  vx = np.zeros((max_steps, N))
  vy = np.zeros((max_steps, N))
  ax = np.zeros((max_steps, N))
  ay = np.zeros((max_steps, N))
  kin_U = np.zeros(max_steps)

  init_particles()

  return

# Initialize positions for all particles
cdef init_particles():
  cdef double theta
  for i in range(N):
    x[0, i] = L * <double>rand.random()
    y[0, i] = L * <double>rand.random()

    theta = <double>(2 * math.pi * rand.random())
    vx[0, i] = v_init * math.cos(theta)
    vy[0, i] = v_init * math.sin(theta)

    theta = <double>(2 * math.pi * rand.random())
    ax[0, i] = a_init * math.cos(theta)
    ay[0, i] = a_init * math.sin(theta)

  return


# Simulation loop
cdef void run_sim(object a):
  cdef int step = 0
  while step < max_steps - 1:
    next = step + 1
    verlet(step)
    a.run(step)
    vel_half_step(step)
    kin_U[step] = kin_energy(next)

    step += 1

  kin_U[step] = kin_energy(next)

  return


# Movement
cdef void verlet(int step):
  cdef int next = step + 1
  cdef double x_new
  cdef double y_new
  cdef int i
  for i in range(N):
    x_new = x[step, i] + vx[step, i] * dt + 0.5 * ax[step, i] * dt_sq
    y_new = y[step, i] + vy[step, i] * dt + 0.5 * ay[step, i] * dt_sq
    vx[next, i] = 0.5 * ax[step, i] * dt
    vy[next, i] = 0.5 * ay[step, i] * dt
    x[next, i] = pbc(x_new)
    y[next, i] = pbc(y_new)

  return


#Update velocity at half steps
cdef void vel_half_step(int step):
  cdef int next = step + 1
  cdef int i
  for i in range(N):
    vx[next, i] += 0.5 * ax[step, i] * dt
    vy[next, i] += 0.5 * ay[step, i] * dt

  return


cdef class accel:
  cdef public double gamma, sigma
  def __init__(self, args=(1., 1., 1.)):
    self.gamma = args[0]
    self.sigma = o_sqrt_dt * math.sqrt(2 * args[0] * args[1] * args[2])
  def run(self, int step):
    cdef double g1, g2, a_x, a_y
    cdef int next = step + 1
    cdef int i
    for i in range(N):
      g1, g2 = gauss()
      a_x = ax[step, i]
      a_y = ay[step, i]
      ax[next, i] = a_x - self.gamma * vx[step, i] + self.sigma * g1
      ay[next, i] = a_y - self.gamma * vy[step, i] + self.sigma * g2



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
    x += L

  return x


# calculate rms velocity
cdef double rms():
  cdef double ms = 0
  cdef int i
  for i in range(N):
    ms += vx[-1, i] ** 2 + vy[-1, i] ** 2

  return np.sqrt(ms)


# calculate average kinetic energy of particles
cpdef double kin_energy(int step):
  cdef double kin = 0
  cdef int i
  for i in range(N):
    kin += 0.5 * (vx[step, i] ** 2 + vy[step, i] ** 2)

  return kin / N


# Set neighbors for each cell
cdef long [:, :] set_neighbors():
  cdef long [:, :] neighbors = np.zeros((cells, 5), dtype=np.int32)
  cdef long [:] naive_neighbors
  cdef int k, i

  for k in range(cells):
    naive_neighbors = np.array([0, 1, lat_size - 1, lat_size, lat_size + 1])
    for i in range(5):
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

    neighbors[k] = naive_neighbors

  return neighbors


cpdef void plot_simple():
  fig, axes = plt.subplots(2, 1, figsize=(8, 6))

  axes[0].plot(kin_U)
  axes[0].plot([0, max_steps], [1, 1], color='gray', linestyle='--')
  axes[0].set_xticks([0, max_steps / 2,  max_steps])
  axes[0].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])
  axes[0].set_xlabel('Time (s)')
  axes[0].set_yticks([0, 1])
  axes[0].set_ylabel(r'Energy $(k_BT)$')
  axes[0].title.set_text('Energy over time')

  cdef double mu = 0.
  cdef double sig = 1.
  count, bins, ignored = axes[1].hist(vx, 80, density=True)
  axes[1].plot(bins, 1/(1 * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mu)**2 / (2 * sig**2) ),
            linewidth=2, color='gray', linestyle='--')
  axes[1].set_xlim([np.percentile(bins, 0.5), np.percentile(bins, 99.5)])
  axes[1].set_xlabel('Velocity')
  axes[1].set_ylabel('Relative Frequency')
  axes[1].title.set_text('Histogram of Particle Velocity')

  fig.tight_layout(pad=3.)
  plt.show()


cpdef void plot_full():
  print('plot_full')
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  fig.suptitle('Energy over time')

  axes[0].plot(kin_U, label=r'$\gamma$ = 1.0')
  axes[1].plot(kin_U, label=r'$k_BT$ = 1.0')
  axes[2].plot(kin_U, label='m = 1.')

  main((2., 1., 1.))
  axes[0].plot(kin_U, label=r'$\gamma$ = 2.0')
  main((0.5, 1., 1.))
  axes[0].plot(kin_U, label=r'$\gamma$ = 0.5')
  axes[0].legend()
  axes[0].set_yticks([0, 1])
  axes[0].set_ylabel(r'Energy $(k_BT)$')

  main((1., 2., 1.))
  axes[1].plot(kin_U, label=r'$k_BT$ = 2.0')
  main((1., 0.5, 1.))
  axes[1].plot(kin_U, label=r'$k_BT$ = 0.5')
  axes[1].legend()
  axes[1].set_yticks([0, 1, 2])
  axes[1].set_ylabel(r'Energy $(k_BT)$')

  main((1., 1., 2.))
  axes[2].plot(kin_U, label='m = 2.0')
  main((1., 1., 0.5))
  axes[2].plot(kin_U, label='m = 0.5')
  axes[2].legend()
  axes[2].set_yticks([0, 1, 2])
  axes[2].set_ylabel(r'Energy $(k_BT)$')
  axes[2].set_xlabel('Time (s)')

  axes[0].set_xticks([0, max_steps / 2,  max_steps])
  axes[0].set_xticklabels([0, max_steps * dt / 2, max_steps * dt])

  axes[0].plot([0, max_steps], [1, 1], color='gray', linestyle='--')
  axes[1].plot([0, max_steps], [1, 1], color='gray', linestyle='--')
  axes[1].plot([0, max_steps], [2, 2], color='gray', linestyle='--')
  axes[1].plot([0, max_steps], [0.5, 0.5], color='gray', linestyle='--')
  axes[2].plot([0, max_steps], [1, 1], color='gray', linestyle='--')
  axes[2].plot([0, max_steps], [2, 2], color='gray', linestyle='--')
  axes[2].plot([0, max_steps], [0.5, 0.5], color='gray', linestyle='--')

  plt.show()
