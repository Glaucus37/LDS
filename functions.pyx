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
  """
    if self.next_p is not None:
      self.next_p.update_pos(dx, dy)
  """

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
