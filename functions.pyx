import sys
import math
from cython import array
import numpy as np
cimport numpy as cnp
import random as rand

# from libc.stdlib cimport rand, srand, RAND_MAX

cython: language_level=3

# Variable definitions
cdef int D = 2
cdef int N = 100
cdef double dt = 1e-2
cdef double t_max = 1e1
cdef double L = 10.
cdef double v_init
cdef double a_init = 2.
cdef double gamma_ = 1.
cdef double kBT = 1.
cdef double m = 1.

cdef double dt_sq = dt ** 2
cdef double o_sqrt_dt = 1 / math.sqrt(dt)
cdef double sigma_ = o_sqrt_dt * math.sqrt(2 * gamma_ * kBT * m)

cdef int max_steps = int(t_max / dt)
cdef int lat_size = 5
cdef long cells = lat_size ** 2


# Array declarations
cdef double [:, :] x = np.zeros((max_steps, N))
cdef double [:, :] y = np.zeros((max_steps, N))
cdef double [:, :] vx = np.zeros((max_steps, N))
cdef double [:, :] vy = np.zeros((max_steps, N))
cdef double [:, :] ax = np.zeros((max_steps, N))
cdef double [:, :] ay = np.zeros((max_steps, N))
cdef double [:] kin_U = np.zeros(max_steps - 1)
# cdef double [:] gauss_vel = np.zeros(2)
# cdef long [:, :] k_neighbors = np.zeros((cells, 5), dtype=np.int32)


cpdef object main():
  rand.seed()
  cdef long [:, :] k_neighbors = set_neighbors()
  
  if len(sys.argv) > 1:
    try:
      v_init = <float>sys.argv[1]
    except TypeError:
      v_init = 1.

  init_particles(v_init)

  run_sim()

  print '\nAverage velocity (RMS): ', rms()

  return kin_U, vx, max_steps


# Initialize positions for all particles
cpdef init_particles(double v_init):
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
cdef void run_sim():
  cdef int step = 0

  while step < max_steps - 1:
    next = step + 1
    verlet(step)
    vel_half_step(step)
    accel(step)
    vel_half_step(step)

    kin_U[step] = kin_energy(step)

    step += 1

  return


# Movement
cdef void verlet(int step):
  cdef int next = step + 1
  cdef double x_new
  cdef double y_new
  for i in range(N):
    x_new = x[step, i] + vx[step, i] * dt + 0.5 * ax[step, i] * dt_sq
    y_new = y[step, i] + vy[step, i] * dt + 0.5 * ay[step, i] * dt_sq
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


cdef void accel(int step):
  cdef double g1, g2, a_x, a_y
  cdef int next = step + 1
  for i in range(N):
    g1, g2 = gauss()
    a_x = ax[step, i]
    a_y = ay[step, i]
    ax[next, i] = a_x - gamma_ * vx[step, i] + sigma_ * g1
    ay[next, i] = a_y - gamma_ * vy[step, i] + sigma_ * g2

  return


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
  for i in range(N):
    ms += vx[-1, i] ** 2 + vy[-1, i] ** 2

  return math.sqrt(ms)


# calculate average kinetic energy of particles
cpdef double kin_energy(int step):
  cdef double kin = 0
  cdef int i
  for i in range(N):
    kin += 0.5 * (vx[step, i] ** 2 + vy[step, i] ** 2)

  return kin / N


# Set neighbors for each cell
cdef long [:, :] set_neighbors():
  neighbors = np.zeros((cells, 5), dtype=np.int32)

  for k in range(cells):
    naive_neighbors = np.array([0, 1, lat_size - 1, lat_size, lat_size + 1])
    naive_neighbors += k

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
