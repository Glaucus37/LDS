import numpy as np
import math
import random as rand
import matplotlib.pyplot as plt
import time

# Section I: classes

class Particle:
  def __init__(self, int n, int L):
    self.index = n
    self.cluster = n

    self.next_p = None

    self.x = L * rand.random()
    self.y = L * rand.random()

  def update_pos(self, double dx, double dy): # NONRECURSIVE
    self.x = PBC(self.x + dx)
    self.y = PBC(self.y + dy)

    if self.next_p is not None:
      self.next_p.update_pos(dx, dy)

  def update_cluster(self, int c): # NONRECURSIVE
    self.cluster = c
    if self.next_p is not None:
      self.next_p.update_cluster(c)


class Cluster:
  def __init__(self, p, sigma):
    v = 2
    a = 1

    self.first_p = p
    self.last_p = p

    self.index = p.index
    self.mass = 1
    self.gamma = 1
    self.kBT = 1
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


cdef double t_max, dt, dt_sq
cdef double gamma, sigma, kBT
cdef double[:] kin_U
cdef int max_steps

cpdef void TimeSetup(double run_t, double run_dt):
  global t_max, dt, max_steps, dt_sq
  global gamma, sigma, kBT
  global max_steps
  global kin_U
  rand.seed()

  t_max = run_t
  dt = run_dt
  max_steps = int(t_max / dt)
  dt_sq = dt**2

  gamma = 1
  kBT = 1
  sigma = np.sqrt(2 * gamma * kBT / dt)

  kin_U = np.zeros(max_steps)


cdef double L, cell_size
cdef int lat_size, cells
cdef long[:, :] k_neighbours

cpdef void InitBoard(double l, int c):
  global L, cell_size
  global lat_size, cells
  global k_neighbours

  L = l
  lat_size = c
  cell_size = l / c
  cells = lat_size**2

  k_neighbours = np.zeros((cells, 5), dtype=np.int32)


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


cpdef void SetNeighbours():
  global k_neighbours
  global lat_size
  cdef int[:] naive_neigbors

  cdef int k, i
  for k in range(cells):
    naive_neighbours = np.array([0, 1, lat_size-1, lat_size, lat_size+1])

    if k % lat_size == 0:
      naive_neighbours[2] += lat_size
    elif k % lat_size == lat_size - 1:
      naive_neighbours[1] -= lat_size
      naive_neighbours[4] -= lat_size
    if k // lat_size == lat_size - 1:
      naive_neighbours[2] -= cells
      naive_neighbours[3] -= cells
      naive_neighbours[4] -= cells

    for i in range(5):
      k_neighbours[k, i] = naive_neighbours[i] + k
      # naive_neighbours[i] += k


cdef int step

cpdef double RunSim():
  global step, max_steps
  global n_clusters
  cdef double t0 = time.perf_counter()
  cdef double t1

  # Accel()

  step = 0
  while step < max_steps - 1 and n_clusters - 1:
    Verlet()
    VelHalfStep()
    Accel()
    VelHalfStep()

    CheckNeighbours() # A.k.a. Particle-Particle Interaction

    if step < max_steps - 1:
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
  global n_clusters
  cdef double g1, g2

  cdef int i
  for i in range(n_clusters):
    g1, g2 = gauss()
    c_list[i].langevin(g1, g2)


cdef Separation(int p_1, int p_2): # PBC
  cdef double d
  p1 = p_list[p_1]
  p2 = p_list[p_2]
  d2 = (p2.x - p1.x)**2 + (p2.y - p1.y)**2
  return np.sqrt(d2)


cdef void CheckNeighbours():
  global N, cells
  global lat_size
  global k_list
  global c_list, p_list
  global cell_size

  cdef int i, j
  for i in range(N, N + cells):
    k_list[i] = -1

  for i in range(N):
    p = p_list[i]
    j = int(p.x / cell_size) + int(p.y / cell_size) * lat_size + N
    k_list[i] = k_list[j]
    k_list[j] = i

  cdef int cell_1, cell_2, particle_1, particle_2
  cdef int c1, c2
  for i in range(cells):
    # for each cell
    cell_1 = i + N
    for j in range(5):
      # for each neighbour of that cell
      particle_1 = k_list[cell_1]
      cell_2 = k_neighbours[i, j] + N
      while particle_1 + 1:
        # for each particle in that cell
        particle_2 = k_list[particle_1]
        while particle_2 + 1:
          # for each neighbouring particle
          if particle_2 < particle_1 or cell_1 != cell_2:
            c1 = p_list[particle_1].cluster
            c2 = p_list[particle_2].cluster

            if c1 != c2:
              if Separation(particle_1, particle_2) <= cell_size:
                if c_list[c1].mass > c_list[c2].mass:
                  JoinClusters(c1, c2)
                else:
                  JoinClusters(c2, c1)
          particle_2 = k_list[particle_2]
        particle_1 = k_list[particle_1]


cdef JoinClusters(int c1, int c2):
  global n_clusters

  first_c = c_list[c1]
  second_c = c_list[c2]
  last_c = c_list[n_clusters - 1]

  #v print('\n\n{} - {} - {}'.format(c1, c2, last_c.index))
  cdef int aux
  if c1 > c2:
    aux = c1
    c1 = c2
    c2 = aux

  # print('{} - {} - {}'.format(c1, c2, last_c.index))
  Momentum(first_c, second_c)

  first_c.last_p.next_p = second_c.first_p
  first_c.last_p = second_c.last_p
  first_c.last_p.next_p = None
  first_c.mass += second_c.mass

  c_list[c1] = first_c
  first_c = c_list[c1] # ????
  first_c.index = c1
  first_c.first_p.update_cluster(c1) # second_c.first_p

  n_clusters -= 1

  if not c2 == n_clusters:
    last_c.index = c2
    last_c.first_p.update_cluster(c2)
    c_list[c2] = last_c

  """
  global k_list
  cdef int i, j
  for i in range(n_clusters):
    c = c_list[i]
    print 'cluster {} ({}):'.format(c.index, c.mass)
    p = c.first_p
    print '\tvx: {}\n\t{}'.format(c.vx, p.index)
    while p.next_p is not None:
      p = p.next_p
      print '\t{}'.format(p.index)
  """


cdef Momentum(first_c, second_c):
  cdef int m1 = first_c.mass
  cdef int m2 = second_c.mass
  cdef double vx1 = first_c.vx
  cdef double vy1 = first_c.vy
  cdef double vx2 = second_c.vx
  cdef double vy2 = second_c.vy

  first_c.vx = (m1 * vx1 + m2 * vx2) / (m1 + m2)
  first_c.vy = (m1 * vy1 + m2 * vy2) / (m1 + m2)


cdef KinEnergy():
  global N, n_clusters
  global c_list
  cdef double kin = 0

  cdef int i
  for i in range(n_clusters):
    c = c_list[i]
    kin += c.mass * (c.vx**2 + c.vy**2)

  kin *= 0.5

  return kin



# Section III: Plots

cpdef void plot():
  global kin_U
  global N, max_steps
  global k_list
  global L, lat_size

  fig, ax = plt.subplots(figsize=(8, 8))

  plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
  cdef int i, j
  cdef double x_i, y_j
  for i in range(lat_size):
    if i:
      x_i = i * cell_size
      for j in range(lat_size):
        if j:
          y_j = j * cell_size
          plt.plot([0, L], [y_j, y_j], color='gray', linestyle='-')
          plt.plot([x_i, x_i], [0, L], color='gray', linestyle='-')

  cdef double[:] x, y
  cdef double s = 4000 / lat_size
  x = np.zeros(N)
  y = np.zeros(N)
  for i in range(N):
    x[i] = p_list[i].x
    y[i] = p_list[i].y
    plt.text(x[i], y[i], p_list[i].index)
  plt.scatter(x, y, s=s)

  plt.xticks(np.linspace(0, L, lat_size + 1))
  plt.yticks(np.linspace(0, L, lat_size + 1))
  plt.xlim(0, 10)
  plt.ylim(0, 10)

  plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
  plt.plot(kin_U, linewidth=1)
  plt.plot([0, max_steps], [1, 1],
    color='gray', linestyle='--')
  plt.text(0, 0.5, kin_U[-1])

  plt.show()
