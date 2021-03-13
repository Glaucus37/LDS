import functions as f
# Particle - 9
# Cluster - 36

def setup(t=1e1, dt=1e-3, L=10, cells=10, n=10):
    f.TimeSetup(t, dt) # 77
    f.InitBoard(L, cells) # 110
    f.InitClusters(n, L) # 129

    t = f.RunSim() # 183
        # Verlet() - 216
        # VelHalfStep() - 248
        # Accel() - 249

        # CheckNeighbors() - 273
        # JoinClusters() - 317

        # KinEnergy() - 324

    print(t)

    f.plot() # 328


if __name__ == '__main__':
    setup()
