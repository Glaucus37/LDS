import functions as f
# Particle - 9
# Cluster - 32

def setup(t=1e2, dt=1e-3, L=10, cells=10, n=10):
    f.TimeSetup(t, dt) # 83
    f.InitBoard(L, cells) # 106
    f.InitClusters(n, L) # 124

    t = f.RunSim() # 182
        # Verlet() - 208
        # VelHalfStep() - 221
        # Accel() - 241

        # CheckNeighbors() - 259
        # JoinClusters() - 302
        # Momentum() - 347

        # KinEnergy() - 347

    print(t)

    f.plot() # 365


if __name__ == '__main__':
    setup()
