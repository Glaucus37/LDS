import numpy as np
import math
import random as rand
import matplotlib.pyplot as plt
import time

# Section I: classes

class Particle:
  def __init__(self, int n, int l):
    self.index = n
    self.cluster = n

    self.next_p = None
    self.prev_p = None

    self.x = L * rand.random()
    self.y = L * rand.random()

  def update_pos(self, double dx, double dy):
    self.x = PBC(self.x + dx)
    self.y = PBC(self.y + dy)

    if self.next_p is not None:
      self.next_p.update_pos(dx, dy)


class Cluster:
  def __init__(self, p, sigma):
    v = 2
    a = 1

    self.first_p = p
    self.last_p = p

    self.index = p.index
    self.mass = 1
    self.gamma = 1
    self.sigma = sigma

    cdef double theta
    theta = 2 * math.pi * rand.random()
    self.vx = v * math.cos(theta)
    self.vy = v * math.sin(theta)

    theta = 2 * math.pi * rand.random()
    self.ax = a * math.cos(theta)
    self.ay = a * math.sin(theta)

  def update_vel(self):
    global dt
    self.vx += 0.5 * self.ax * dt
    self.vy += 0.5 * self.ay * dt

  def langevin(self, double g1, double g2):
    self.ax = -self.gamma * self.mass * self.vx + self.sigma * g1
    self.ay = -self.gamma * self.mass * self.vy + self.sigma * g2



# Section II: functions

cdef PBC(double x):
  global L

  if x < 0:
    x += L
  elif x >= L:
    x -= L

  return x


cdef double t_max, dt, dt_sq, o_sqrt_dt
cdef double gamma, sigma
cdef double[:] kin_U
cdef int max_steps

cpdef void TimeSetup(double run_t, double run_dt):
  global t_max, dt, max_steps, dt_sq, o_sqrt_dt
  global gamma, sigma
  global max_steps
  global kin_U
  rand.seed()

  t_max = run_t
  dt = run_dt
  max_steps = int(t_max / dt)
  dt_sq = dt**2
  o_sqrt_dt = 1 / np.sqrt(dt)

  gamma = 1
  sigma = o_sqrt_dt * np.sqrt(2 * 1)

  kin_U = np.zeros(max_steps)


cdef double L, cell_size
cdef int lat_size, cells
cdef long[:, :] k_neighbors

cpdef void InitBoard(double l, int c):
  global L, cell_size
  global lat_size, cells
  global k_neighbors

  L = l
  lat_size = c
  cell_size = l / c
  cells = lat_size**2

  k_neighbors = np.zeros((cells, 5), dtype=np.int32)


cdef int N, n_clusters
cdef long[:] k_list
p_list = []
c_list = []

cpdef void InitClusters(int n, double l):
  global N, n_clusters
  N = n
  n_clusters = n

  global k_list
  k_list = np.zeros(N + cells, dtype=np.int32)

  cdef double vx_cm = 0
  cdef double vy_cm = 0

  global L, sigma
  L = l
  global p_list
  global c_list

  cdef int i
  p_list = [Particle(i, L) for i in range(N)]
  c_list = [Cluster(p_list[i], sigma) for i in range(N)]

  for i in range(N):
    vx_cm += c_list[i].vx
    vy_cm += c_list[i].vy

  vx_cm /= N
  vy_cm /= N

  for i in range(N):
    c_list[i].vx += vx_cm
    c_list[i].vy += vy_cm


cdef void SetNeighbors():
  global k_neighbors
  global lat_size
  cdef int[:] naive_neigbors

  cdef int k, i
  for k in range(cells):
    naive_neighbors = np.array([0, 1, lat_size-1, lat_size, lat_size+1])
    for i in range(5):
      naive_neighbors[i] += k

  if k % lat_size == 0:
    naive_neighbors[2] += lat_size
  elif k % lat_size == lat_size - 1:
    naive_neighbors[1] -= lat_size
    naive_neighbors[4] -= lat_size
  if k // lat_size == lat_size - 1:
    naive_neighbors[2] -= cells
    naive_neighbors[3] -= cells
    naive_neighbors[4] -= cells

  k_neighbors[k] = naive_neighbors


cdef int step, next

cpdef double RunSim():
  global step, next, max_steps
  cdef double t0 = time.perf_counter()
  cdef double t1

  step = 0
  while step < max_steps - 1:
    next = step + 1

    Verlet()
    VelHalfStep()
    Accel()
    VelHalfStep()


    # NeighbouringCells()

    kin_U[step] = KinEnergy()

    step += 1

  kin_U[step] = KinEnergy()

  t1 = time.perf_counter()

  return t1 - t0


cdef void Verlet():
  global N, n_clusters
  global dt, dt_sq
  global c_list, p_list
  cdef double dx, dy

  cdef int i
  for i in range(n_clusters):
    dx = c_list[i].vx * dt + 0.5 * c_list[i].ax * dt_sq
    dy = c_list[i].vy * dt + 0.5 * c_list[i].ay * dt_sq
    c_list[i].first_p.update_pos(dx, dy)


cdef VelHalfStep():
  global n_clusters

  cdef int i
  for i in range(n_clusters):
    c_list[i].update_vel()


# generate (and return) a pair of normally distributed numbers
cdef (double, double) gauss(double s_d=1.):
  cdef double fac, v1, v2
  cdef double r_sq = 0.
  while r_sq <= 0. or r_sq > 1.:
    v1 = 2. * rand.random() - 1.
    v2 = 2. * rand.random() - 1.
    r_sq = v1 ** 2 + v2 ** 2
  fac = s_d * np.sqrt(-2. * np.log(r_sq) / r_sq)
  return (v1 * fac, v2 * fac)


cdef Accel():
  global N
  cdef double g1, g2

  cdef int i
  for i in range(N):
    g1, g2 = gauss()
    c_list[i].langevin(g1, g2)


cdef void SetKList():
  global N, cells
  global lat_size
  global k_list
  global p_list

  cdef int i, k
  for i in range(N, cells):
    k_list[i] = -1

    for i in range(N):
      p = p_list[i]
      k = int(p.x / cell_size) + int(p.y / cell_size) * lat_size + N
      k_list[i] = k_list[k]
      k_list[k] = i


cdef NeighbouringCells():
  global N, n_clusters
  global c_list, p_list
  global k_list

  cdef int i, j, m1, m2, j1, j2
  for i in range(cells):
    m1 = k + N
    for j in range(5):
      m2 = k_neighbors[k, i] + N
      j1 = k_list[m1]
      while j1 >= 0:
        j2 = k_list[m2]
        while j2 >= 0:
          if j2 < j1 or m1 != m2:
            c1 = p_list[j1].cluster
            c2 = p_list[j2].cluster
            c1 != c2:
              if c_list[c1].mass < c_list[c2].mass
                JoinClusters(c2, c1)



cdef JoinClusters(int m1, int m2):



cdef KinEnergy():
  global N, n_clusters
  global c_list
  cdef double kin = 0

  cdef int i
  for i in range(n_clusters):
    c = c_list[i]
    kin += c.mass * (c.vx**2 + c.vy**2)

  kin *= 0.5 / N

  return kin



# Section III: Plots

cpdef void plot():
  global kin_U
  global N, max_steps
  global k_list
  print(N)

  fig, ax = plt.subplots(figsize=(8, 8))

  plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
  cdef double[:] x, y
  cdef int i, j
  x = np.zeros(N)
  y = np.zeros(N)
  for i in range(N):
    x[i] = p_list[i].x
    y[i] = p_list[i].y
  for i in range(N):
    print(x[i], y[i])
  plt.scatter(x, y, s=400)

  cdef double x_i, y_j
  for i in range(lat_size):
    if i:
      x_i = i * cell_size
      for j in range(lat_size):
        if j:
          y_j = j * cell_size
          plt.plot([0, L], [y_j, y_j], color='gray', linestyle='-')
          plt.plot([x_i, x_i], [0, L], color='gray', linestyle='-')

  global L, lat_size
  plt.xticks(np.linspace(0, L, lat_size + 1))
  plt.yticks(np.linspace(0, L, lat_size + 1))
  plt.xlim(0, 10)
  plt.ylim(0, 10)

  plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
  plt.plot(kin_U, linewidth=1)
  plt.plot([0, max_steps], [1, 1],
    color='gray', linestyle='--')

  plt.show()
