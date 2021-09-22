import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time
import cython


cdef class Board:
  cdef public int board_length, n_cells, max_steps, gamma, sigma, step
  cdef public int n_clusters, n_particles
  cdef public double max_t, dt, dt_sq, kBT
  cdef public long[:] k_list
  cdef public long[:, :] k_neighbours
  cdef public double[:] kin_U
  cdef public list p_list, c_list

  # Initialization
  def __init__(self, int n=2, int l=10, double t=1e1, double dt=1e-1):
    self.board_length = l
    self.n_cells = l ** 2
    self.max_t = t
    self.dt = dt
    self.TimeSetup()
    self.InitClusters(n)
    self.InitNeighbours()

  def TimeSetup(self):
    rand.seed(37)
    self.max_steps = int(self.max_t / self.dt)
    self.kin_U = np.zeros(self.max_steps, dtype=float)
    self.dt_sq = self.dt ** 2
    self.gamma = 1
    self.kBT = 1
    self.sigma = np.sqrt(2 * self.gamma * self.kBT / self.dt)

  def InitClusters(self, int n=2):
    density = 1e-1
    N = int(n * density * self.n_cells / np.pi)
    self.n_clusters = N
    self.n_particles = N
    self.k_list = np.zeros(N + self.n_cells, dtype=int)

    p_list = [Particle(i, self) for i in range(N)]
    self.p_list = p_list
    c_list = [Cluster(i, self) for i in range(N)]
    self.c_list = c_list

    cdef int i
    vx_cm = 0
    vy_cm = 0
    for i in range(N):
      vx_cm += c_list[i].vx
      vy_cm += c_list[i].vy
    vx_cm /= N
    vy_cm /= N
    for i in range(N):
      c_list[i].vx -= vx_cm
      c_list[i].vy -= vy_cm

  def InitNeighbours(self):
    L = self.board_length
    L_2 = self.n_cells
    self.k_neighbours = np.zeros((L_2, 5), dtype=int)
    cdef int i, j
    for i in range(L_2):
      naive_neighbours = np.array([0, 1, L-1, L, L+1])
      if i % L == 0:
        naive_neighbours[2] += L
      elif i % L == L - 1:
        naive_neighbours[1] -= L
        naive_neighbours[4] -= L
      if i // L == L - 1:
        naive_neighbours[2] -= L_2
        naive_neighbours[3] -= L_2
        naive_neighbours[4] -= L_2
      for j in range(5):
        self.k_neighbours[i, j] = naive_neighbours[j] + i

  # Neighbours Logic
  def Separation(self, int i, int j):
    p1 = self.p_list[i]
    p2 = self.p_list[j]
    dx = abs(p2.x - p1.x)
    if dx > 0.5 * self.board_length:
      dx -= self.board_length
    dy = abs(p2.y - p1.y)
    if dy > 0.5 * self.board_length:
      dy -= self.board_length
    d2 = dx**2 + dy**2
    return np.sqrt(d2) <= 1

  def CheckNeighbours(self):
    cdef int i, j, cell_1, cell_2, particle_1, particle_2, c1, c2, m1, m2
    for i in range(self.n_particles, self.n_particles + self.n_cells):
      self.k_list[i] = -1
    for i in range(self.n_particles):
      p = self.p_list[i]
      j = int(p.x) + int(p.y) * self.board_length + self.n_particles
      if j >= self.n_particles + self.n_cells:
        j -= self.n_cells
      self.k_list[i] = self.k_list[j]
      self.k_list[j] = i
    for i in range(self.n_cells):
      cell_1 = i + self.n_particles
      particle_1 = self.k_list[cell_1]
      while particle_1 + 1:
        for j in range(5):
          cell_2 = self.k_neighbours[i, j] + self.n_particles
          particle_2 = self.k_list[cell_2]
          while particle_2 + 1:
            if particle_2 < particle_1 or cell_1 != cell_2:
              c1 = self.p_list[particle_1].cluster
              c2 = self.p_list[particle_2].cluster
              if c1 != c2 and self.Separation(particle_1, particle_2):
                m1 = self.c_list[c1].mass
                m2 = self.c_list[c2].mass
                self.JoinClusters(c1, c2) if m1 > m2 else self.JoinClusters(c2, c1)
            particle_2 = self.k_list[particle_2]
        particle_1 = self.k_list[particle_1]

  def JoinClusters(self, int c1, int c2):
    cdef Cluster first_c, second_c, last_c
    first_c = self.c_list[c1]
    second_c = self.c_list[c2]
    last_c = self.c_list[self.n_clusters - 1]
    cdef int aux
    if c1 > c2:
      aux = c1
      c1 = c2
      c2 = aux
    self.Momentum(first_c, second_c)

    first_c.last_p.next_p = second_c.first_p
    first_c.last_p = second_c.last_p
    first_c.mass += second_c.mass

    self.c_list[c1] = first_c
    first_c.UpdateCluster(c1)
    self.n_clusters -= 1

    if not c2 == self.n_clusters:
      self.c_list[c2] = last_c
      last_c.UpdateCluster(c2)

  # Generic Functions
  def Momentum(self, Cluster first_c, Cluster second_c):
    cdef int m1, m2
    cdef double vx1, vy1, vx2, vy2
    m1 = first_c.mass
    m2 = second_c.mass
    vx1 = first_c.vx
    vy1 = first_c.vy
    vx2 = second_c.vx
    vy2 = second_c.vy
    first_c.vx = (m1 * vx1 + m2 * vx2) / (m1 + m2)
    first_c.vy = (m1 * vy1 + m2 * vy2) / (m1 + m2)

  def PBC(self, double x):
    if x < 0:
      x += self.board_length
    elif x >= self.board_length:
      x -= self.board_length
    return x

  def Gauss(self, double s_d=1.):
    cdef double r_sq, v1, v2, fac
    r_sq = 0.
    while r_sq <= 0. or r_sq > 1.:
      v1 = 2. * rand.random() - 1.
      v2 = 2. * rand.random() - 1.
      r_sq = v1 ** 2 + v2 ** 2
    fac = s_d * np.sqrt(-2 * np.log(r_sq) / r_sq)
    return (v1 * fac, v2 * fac)

  # Iterated Functions
  def ClusterSimStep(self):
    cdef Cluster c
    for c in self.c_list[:self.n_clusters]:
      c.UpdatePos()
      c.UpdateVel()
      c.UpdateAcc()
      c.UpdateVel()
    self.kin_U[self.step] = self.CalculateEnergy()
    self.CheckNeighbours()

  def CalculateEnergy(self):
    cdef double energy = 0.
    for c in self.c_list[:self.n_clusters]:
      energy += c.KinEnergy()
    return 0.5 * energy

  # Plots
  def PlotBoard(self):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    L = self.board_length
    for p in self.p_list:
      self.DrawCircle((p.x, p.y))
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.show()

  def PlotEnergy(self):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    max_energy = np.amax(self.kin_U)
    plt.plot([-1, self.step], [0, 0], 'k', linestyle='-', linewidth=0.2)
    plt.plot([0, 0], [-1, max_energy*1.1], 'k', linestyle='-', linewidth=0.2)
    plt.plot([-1, self.step], [self.kBT, self.kBT], 'r', linestyle='-', linewidth=0.5)
    plt.plot(self.kin_U[:self.step])
    plt.ylabel(r'Energy ($k_BT$)')
    plt.yticks([0, 1], ['0', '1'])
    plt.xlabel('time (computational units)')
    plt.xticks([], [])

  def DrawCircle(self, pos):
    circle = plt.Circle(pos, 0.5)
    plt.gcf().gca().add_artist(circle)

  # Run Simulation
  def RunSim(self):
    cdef double t0
    t0 = time.time()
    self.step = 0
    cdef Cluster c
    for c in self.c_list:
      c.UpdateAcc()
    while self.step < self.max_steps - 1 and self.n_clusters - 1:
      self.ClusterSimStep()
      self.step += 1
    self.kin_U[self.step] = self.CalculateEnergy()
    print(f'Run Duration: {round(time.time() - t0, 3)}s')
    self.PlotEnergy()
    self.PlotBoard()


cdef class Particle:
  cdef Board board
  cdef public int index, cluster
  cdef public Particle next_p
  cdef public double x, y

  def __init__(self, int i, Board board):
    self.board = board
    self.index = i
    self.cluster = i
    self.next_p = None

    cdef double n_sqrt
    n_sqrt = np.sqrt(board.n_particles)
    cdef int step
    step = int(board.board_length / n_sqrt)

    self.x = board.PBC(step * (i % n_sqrt))
    self.y = board.PBC(step * (i // n_sqrt))


cdef class Cluster:
  cdef Board board
  cdef public int index, mass
  cdef public Particle first_p, last_p
  cdef public double vx, vy, ax, ay

  def __init__(self, int i, Board board):
    self.board = board
    self.index = i
    self.mass = 1
    cdef Particle p
    p = board.p_list[i]
    self.first_p = p
    self.last_p = p

    cdef double theta
    theta = 2 * np.pi * rand.random()
    cdef double v0 = 1
    self.vx = v0 * np.cos(theta)
    self.vy = v0 * np.sin(theta)

  def UpdatePos(self):
    cdef double dx, dy
    dx = self.vx * self.board.dt + 0.5 * self.ax * self.board.dt_sq
    dy = self.vy * self.board.dt + 0.5 * self.ay * self.board.dt_sq
    cdef Particle p
    p = self.first_p

    while p is not None:
      p.x = self.board.PBC(p.x + dx)
      p.y = self.board.PBC(p.y + dy)
      p = p.next_p

  def UpdateVel(self):
    self.vx += 0.5 * self.ax * self.board.dt
    self.vy += 0.5 * self.ay * self.board.dt

  def UpdateAcc(self):
    cdef double g1, g2
    g1, g2 = self.board.Gauss()
    self.ax = -self.board.gamma * self.mass * self.vx + self.board.sigma * g1
    self.ay = -self.board.gamma * self.mass * self.vy + self.board.sigma * g2

  def UpdateCluster(self, int c):
    self.index = c
    cdef Particle p
    p = self.first_p
    while p is not None:
      p.cluster = c
      p = p.next_p

  def ListCluster(self):
    cdef Particle p
    p = self.first_p
    while p is not None:
      print(p.index, end='')
      p = p.next_p
    print('')

  def KinEnergy(self):
    return self.mass * (self.vx**2 + self.vy**2)
