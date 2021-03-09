import numpy as np
import math
import random as rand
import time

# Section I: classes

cdef class Particle:
  cdef public int index, cluster
  cdef public double x, y
  cdef public Particle next_p

  def __init__(self, int n, double L):
    self.index = n
    self.cluster = n

    self.next_p = None

    self.x = L * rand.random()
    self.y = L * rand.random()

  def update_pos(self, double dx, double dy):
    self.x += dx
    self.y += dy

    if self.next_p is not None:
      self.next_p.update_pos(dx, dy)


cdef class Cluster:
  cdef public int index, mass
  cdef public double vx, vy, ax, ay
  cdef public double gamma, sigma
  cdef public Particle first_p

  def __init__(self, Particle p):
    cdef double v = 2
    cdef double a = 1

    self.first_p = p

    self.index = p.index
    self.mass = 1
    self.gamma = 1
    self.sigma = 1

    cdef double theta
    theta = 2 * math.pi * rand.random()
    self.vx = v * math.cos(theta)
    self.vy = v * math.sin(theta)

    theta = 2 * math.pi * rand.random()
    self.ax = a * math.cos(theta)
    self.ay = a * math.sin(theta)

    def update_vel(self):
      self.vx += 0.5 * self.ax * dt
      self.vy += 0.5 * self.ay * dt

    def langevin(self, double g1, double g2):
      self.ax += -self.gamma * self.mass * self.vx + self.sigma * g1
      self.ay += -self.gamma * self.mass * self.vy + self.sigma * g2


# Section II: functions

cdef double t_max, dt, dt_sq, o_sqrt_dt
cdef double[:] kin_U

cpdef void time_setup(double run_t, double run_dt):
  global t_max, dt, max_steps, dt_sq, o_sqrt_dt
  global kin_U

  t_max = run_t
  dt = run_dt
  max_steps = int(t_max / dt)
  dt_sq = dt**2
  o_sqrt_dt = 1 / np.sqrt(dt)

  kin_U = np.zeros(max_steps)


cdef double L, cell_size
cdef int lat_size, cells

cpdef void init_board(double l, int c):
  global L, cell_size
  global lat_size, cells

  L = l
  lat_size = c
  cell_size = l / c
  cells = lat_size**2

  """
  set_neighbors()

cdef long [:, :] k_neighbors = np.zeros((cells, 5), dtype=np.int32)

cdef void set_neighbors():
  # any cell always has 9 neighbors (including itself), 4 of
    # which will have it as their neighbor
  # as such, we can safely assign 5 'forward' neighbors to each cell
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
"""

cdef Particle[:] p_list # particle list
cdef Cluster[:] c_list # cluster list
cdef int N

cpdef void init_clusters(int n):
  global N
  N = n

  cdef double vx_cm = 0
  cdef double vy_cm = 0

  global L
  global p_list, c_list
  cdef int i
  p_list = [Particle(i, L) for i in range(n)]
  c_list = [Cluster(p_list[i]) for i in range(n)]

  for i in range(n):
    vx_cm += c_list[i].vx
    vy_cm += c_list[i].vy

  vx_cm /= n
  vy_cm /= n

  for i in range(n):
    c_list[i].vx -= vx_cm
    c_list[i].vy -= vy_cm


cdef int step, next

cpdef double run_sim():
  global step, next, max_steps
  cdef double t_0 = time.perf_counter()
  cdef double t_1

  step = 0
  while step < max_steps - 1:
    next = step + 1

    verlet()
    vel_half_step()
    accel()
    vel_half_step()

    # kin_U[step] = kin_energy()

    step += 1

  # kin_U[step] = kin_energy()


cdef void verlet():
  global N
  global dt, dt_sq
  global c_list, p_list
  cdef double dx, dy

  cdef int i
  for i in range(N):
    dx = c_list[i].vx * + 0.5 * c_list[i].ax * dt_sq
    dy = c_list[i].vy * + 0.5 * c_list[i].ay * dt_sq
    c_list[i].first_p.update_pos(dx, dy)


cdef void accel():
  global N

  cdef double g1, g2

  cdef int i
  for i in range(N):
    g1, g2 = gauss()
    c_list[i].langevin(g1, g2)


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


cdef void vel_half_step():
  global N

  cdef int i
  for i in range(N):
    p_list[i].update_vel()
