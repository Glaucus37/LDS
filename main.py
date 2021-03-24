import functions as f
# Particle - 9
# Cluster - 32

def setup(t=1e3, dt=1e-4, L=10, cells=10, n=20):
    f.TimeSetup(t, dt) # 83
    f.InitBoard(L, cells) # 107
    f.InitClusters(n, L) # 125
    f.SetNeighbours() # 157

    t = f.RunSim() # 183
        # Verlet() - 208
        # VelHalfStep() - 221
        # Accel() - 241

        # CheckNeighbors() - 259
        # JoinClusters() - 305
        # Momentum() - 349

        # KinEnergy() - 363

    print(t)

    f.plot() # 381


if __name__ == '__main__':
    setup()
