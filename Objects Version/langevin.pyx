import math
import random as rand


cdef double vx_cm = 0.
cdef double vy_cm = 0.
cdef double L = 10.

cdef int [:] p_list # particle list
cdef int [:] c_list # cluster list


def create_clusters(int n):
  global vx_cm, vy_cm

  cdef int i
  for i in range(n):
    print(type(i))
    p_list[i] = Particle(n=i, v=1.5)
    c_list[i] = Cluster(p_list[i])

    vx_cm += p_list[i].vx
    vy_cm += p_list[i].vy

  vx_cm /= n
  vy_cm /= n

  for i in range(n):
    p_list[i].vx -= vx_cm
    p_list[i].vy -= vy_cm





cdef class Cluster:

  def __init__(self, Particle part):
    self.first_p = part
    self.mass = 1
    self.vx = part.vx
    self.vy = part.vy



cdef class Particle:
  cdef public double index, cluster, x, y, vx, vy, ax, ay

  def __init__(self, int n, double v=2., double a=1.):
    self.index = n
    self.cluster = n

    global L
    self.x = L * rand.random()
    self.y = L * rand.random()

    cdef double theta
    theta = 2 * math.pi * rand.random()
    self.vx = v * math.cos(theta)
    self.vy = v * math.sin(theta)

    theta = 2 * math.pi * rand.random()
    self.ax = a * math.cos(theta)
    self.ay = a * math.sin(theta)
