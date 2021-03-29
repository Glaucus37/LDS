import functions as f
# Particle - 9
# Cluster - 32

def setup(t=1e2, dt=1e-3, L=10, cells=10, n=10):
    f.TimeSetup(t, dt) # 83
    f.InitBoard(L, cells) # 110
    f.InitClusters(n, L) # 125
    f.SetNeighbours() # 157


if __name__ == '__main__':
    setup()

    t = f.RunSim() # 184
        # Verlet() - 217
        # VelHalfStep() - 221
        # Accel() - 245

        # CheckNeighbors() - 259
        # JoinClusters() - 305
        # Momentum() - 349

        # KinEnergy() - 363

    print(t)

    f.plot() # 381
