import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time


class Board:
    def __init__(self, n=2, l=10, t=1e1, dt=1e-1):
        self.board_length = l
        self.n_cells = l ** 2
        self.cell_size = 1
        self.max_t = t
        self.dt = dt
        self.TimeSetup()
        self.InitClusters(n)
        self.InitNeighbours()

    def TimeSetup(self):
        rand.seed(37)
        self.max_steps = self.max_t / self.dt
        self.dt_sq = self.dt ** 2
        self.gamma = 1
        self.kBT = 1
        self.sigma = np.sqrt(2 * self.gamma * self.kBT / self.dt)

    def InitClusters(self, n=2):
        density = 1e-1
        N = int(n * density * self.n_cells / np.pi)
        self.n_clusters = N
        self.n_particles = N
        self.k_list = np.zeros(N + self.n_cells, dtype=np.int32)

        p_list = [Particle(i, self) for i in range(N)]
        self.p_list = p_list
        c_list = [Cluster(i, self) for i in range(N)]
        self.c_list = c_list

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
        self.k_neighbours = np.zeros((L_2, 5), dtype=np.int32)
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

    def Separation(self, i, j):
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

    def JoinClusters(self, c1, c2):
        first_c = self.c_list[c1]
        second_c = self.c_list[c2]
        last_c = self.c_list[self.n_clusters - 1]
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

    def Momentum(self, first_c, second_c):
        m1 = first_c.mass
        m2 = second_c.mass
        vx1 = first_c.vx
        vy1 = first_c.vy
        vx2 = second_c.vx
        vy2 = second_c.vy
        first_c.vx = (m1 * vx1 + m2 * vx2) / (m1 + m2)
        first_c.vy = (m1 * vy1 + m2 * vy2) / (m1 + m2)

    def PBC(self, x):
        if x < 0:
            x += self.board_length
        elif x >= self.board_length:
            x -= self.board_length
        return x

    def Gauss(self, s_d=1.):
        r_sq = 0.
        while r_sq <= 0. or r_sq > 1.:
            v1 = 2. * rand.random() - 1.
            v2 = 2. * rand.random() - 1.
            r_sq = v1 ** 2 + v2 ** 2
        fac = s_d * np.sqrt(-2 * np.log(r_sq) / r_sq)
        return (v1 * fac, v2 * fac)

    def ClusterSimStep(self):
        for c in self.c_list[:self.n_clusters]:
            c.UpdatePos()
            c.UpdateVel()
            c.UpdateAcc()
            c.UpdateVel()
        self.CheckNeighbours()

    def PlotBoard(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
        L = self.board_length
        for p in self.p_list:
            self.DrawCircle((p.x, p.y))
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.show()

    def DrawCircle(self, pos):
        circle = plt.Circle(pos, 0.5)
        plt.gcf().gca().add_artist(circle)

    def ListAllClusters(self):
        for c in self.c_list[:self.n_clusters]:
            c.ListCluster()

    def RunSim(self):
        t0 = time.time()
        self.step = 0
        for c in self.c_list:
            c.UpdateAcc()
        while self.step < self.max_steps - 1 and self.n_clusters - 1:
            self.ClusterSimStep()
            self.step += 1
        print(f'Run Duration: {round(time.time() - t0, 3)}s')
        self.PlotBoard()


class Particle:
    def __init__(self, i, board):
        self.board = board
        self.index = i
        self.cluster = i
        self.next_p = None

        n_sqrt = np.sqrt(board.n_particles)
        step = int(board.board_length / n_sqrt)

        self.x = board.PBC(step * (i % n_sqrt))
        self.y = board.PBC(step * (i // n_sqrt))


class Cluster:
    def __init__(self, i, board):
        self.board = board
        self.index = i
        p = board.p_list[i]
        self.first_p = p
        self.last_p = p
        self.mass = 1

        theta = 2 * np.pi * rand.random()
        self.vx = np.cos(theta)
        self.vy = np.sin(theta)

    def UpdatePos(self):
        dx = self.vx * self.board.dt + 0.5 * self.ax * self.board.dt_sq
        dy = self.vy * self.board.dt + 0.5 * self.ay * self.board.dt_sq
        p = self.first_p

        while p is not None:
            p.x = self.board.PBC(p.x + dx)
            p.y = self.board.PBC(p.y + dy)
            p = p.next_p

    def UpdateVel(self):
        self.vx += 0.5 * self.ax * self.board.dt
        self.vy += 0.5 * self.ay * self.board.dt

    def UpdateAcc(self):
        g1, g2 = self.board.Gauss()
        self.ax = -self.board.gamma * self.mass * self.vx + self.board.sigma * g1
        self.ay = -self.board.gamma * self.mass * self.vy + self.board.sigma * g2

    def UpdateCluster(self, c):
        self.index = c
        p = self.first_p
        while p is not None:
            p.cluster = c
            p = p.next_p

    def ListCluster(self):
        p = self.first_p
        while p is not None:
            print(p.index, end='')
            p = p.next_p
        print('')


if __name__ == '__main__':
    board = Board(t=1e3, dt=1e-2, l=100)
    board.RunSim()
