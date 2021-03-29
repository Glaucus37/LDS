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

    self.x = 4 + n
    self.y = 5 + int(n / 2)

    # self.x = L * rand.random()
    # self.y = L * rand.random()


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

    if p.index % 2:
      self.vx = v
    else:
      self.vx = -v

    self.vy = 0

    self.ax = 0
    self.ay = 0

    """
    cdef double theta
    theta = 2 * math.pi * rand.random()
    self.vx = v * math.cos(theta)
    self.vy = v * math.sin(theta)
    """

  def update_pos(self, double dx, double dy):
    p = self.first_p
    next = True
    while next:
      p.x = PBC(p.x + dx)
      p.y = PBC(p.y + dy)
      p = p.next_p
      if p is None:
        next = False

  def update_vel(self):
    global dt
    self.vx += 0.5 * self.ax * dt
    self.vy += 0.5 * self.ay * dt

  def update_cluster(self, int c):
    p = self.first_p
    next = True
    while next:
      p.cluster = c
      p = p.next_p
      if p is None:
        next = False

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

cpdef void InitClusters_2(double l):
  global N, n_clusters
  N = 10
  n_clusters = N

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


cdef int step

cpdef double RunSim():
  global step, max_steps
  global n_clusters
  global dx_max
  cdef double t0 = time.perf_counter()
  cdef double t1
  cdef int join = 0

  Accel()

  step = 0
  while step < max_steps - 1 and n_clusters - 1:
    Verlet()
    VelHalfStep()
    Accel()
    VelHalfStep()

    joined = CheckNeighbours() # A.k.a. Particle-Particle Interaction
    """
    if not joined:
      if step - join > 100:
        join = step
        CheckJoin()
    """

    if step < max_steps - 1:
      kin_U[step] = KinEnergy()

    # if not step % 1000:
      # PlotClusters()

    step += 1

  kin_U[step] = KinEnergy()
  print('dx_max: {}'.format(dx_max))

  t1 = time.perf_counter()
  return t1 - t0


cpdef double RunAltSim():
  global step, max_steps
  global n_clusters
  global dx_max
  cdef double t0 = time.perf_counter()
  cdef double t1

  # Accel()

  step = 0
  while step < max_steps - 1 and n_clusters - 1:
    Verlet()
    VelHalfStep()
    # Accel()
    VelHalfStep()

    joined = CheckNeighbours() # A.k.a. Particle-Particle Interaction

    if step < max_steps - 1:
      kin_U[step] = KinEnergy()

    step += 1

  kin_U[step] = KinEnergy()
  print('dx_max: {}'.format(dx_max))

  t1 = time.perf_counter()
  return t1 - t0


cdef double dx_max = 0

cdef void Verlet():
  global N, n_clusters
  global dt, dt_sq
  global c_list, p_list
  global dx_max
  cdef double dx, dy

  cdef int i
  for i in range(n_clusters):
    dx = c_list[i].vx * dt + 0.5 * c_list[i].ax * dt_sq
    if abs(dx) > dx_max:
      dx_max = abs(dx)
    dy = c_list[i].vy * dt + 0.5 * c_list[i].ay * dt_sq
    c_list[i].update_pos(dx, dy)


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
  cdef double dx, dy, d2
  global L

  p1 = p_list[p_1]
  p2 = p_list[p_2]

  dx = abs(p2.x - p1.x)
  if dx > 0.5 * L:
    dx -= L
  dy = abs(p2.y - p1.y)
  if dy > 0.5 * L:
    dy -= L

  d2 = (dx)**2 + (dy)**2
  return np.sqrt(d2)


cdef CheckNeighbours(debug=False):
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

  cdef double sep
  cdef int joined = 0
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
              sep = Separation(particle_1, particle_2)
              if sep <= cell_size:
                PlotClusters(0)
                joined = 1
                if c_list[c1].mass > c_list[c2].mass:
                  JoinClusters(c1, c2)
                else:
                  JoinClusters(c2, c1)
                PlotClusters(1)
            else:
              if CheckJoin():
                return [particle_1, particle_2, cell_1, cell_2]

          particle_2 = k_list[particle_2]
        particle_1 = k_list[particle_1]

  return joined


cdef CheckJoin():
  global cell_size
  global colors

  cdef double sep

  cdef int i
  for p in p_list:
    for q in p_list:
      if p.index < q.index and not p.cluster == q.cluster:
        try:
          sep = Separation(p.index, q.index)
          if sep <= cell_size:
            print(round(sep, 2), colors[p.cluster], colors[q.cluster], sep=' - ')
            args = CheckNeighbours(debug=True)
            print(args)
            PlotClusters(2)
            return True
        except TypeError:
          print(sep)

          return False


cdef void JoinClusters(int c1, int c2):
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
  first_c.mass += second_c.mass

  c_list[c1] = first_c
  first_c.index = c1
  first_c.update_cluster(c1) # second_c

  n_clusters -= 1

  if not c2 == n_clusters:
    last_c.index = c2
    last_c.update_cluster(c2)
    c_list[c2] = last_c

  # PlotClusters()

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

colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'orange', 'fuchsia', 'greenyellow']

def PlotClusters(joined=False):
  global kin_U
  global N, max_steps
  global k_list
  global L, lat_size
  global step

  fig, ax = plt.subplots(figsize=(8, 8))

  cdef int i, j
  cdef double x_i, y_j
  cdef double R = cell_size / 2
  cdef int n = 64
  cdef double[:] t = np.linspace(0, 2*np.pi, n+1)
  plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
  plt.title('after' if joined else 'before')
  if joined == 2:
    plt.text(5, 5, '!!')

  for i in range(lat_size):
    if i:
      x_i = i * cell_size
      for j in range(lat_size):
        if j:
          y_j = j * cell_size
          plt.plot([0, L], [y_j, y_j], color='gray', linestyle='-')
          plt.plot([x_i, x_i], [0, L], color='gray', linestyle='-')

  for p in p_list:
    x = R * np.cos(t) + p.x
    y = R * np.sin(t) + p.y
    plt.plot(x, y, color=colors[p.cluster])

  plt.xticks(np.linspace(0, L, lat_size + 1))
  plt.yticks(np.linspace(0, L, lat_size + 1))
  plt.xlim(0, 10)
  plt.ylim(0, 10)

  plt.show()


cpdef void plot():
  global kin_U
  global N, max_steps
  global k_list
  global L, lat_size
  global step

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

  colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'orange', 'fuchsia', 'greenyellow']
  cdef double R = cell_size / 2
  cdef int n = 64
  cdef double[:] t = np.linspace(0, 2*np.pi, n+1)
  for p in p_list:
    x = R * np.cos(t) + p.x
    y = R * np.sin(t) + p.y
    plt.plot(x, y, color=colors[p.cluster])

  plt.xticks(np.linspace(0, L, lat_size + 1))
  plt.yticks(np.linspace(0, L, lat_size + 1))
  plt.xlim(0, 10)
  plt.ylim(0, 10)

  plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
  plt.plot(kin_U[:step], linewidth=1)
  plt.plot([0, step], [1, 1],
    color='gray', linestyle='--')

  plt.show()
