import functions as f
# Particle - 9
# Cluster - 27

def setup(t=1e1, dt=1e-3, L=10, cells=10, n=10):
    f.TimeSetup(t, dt) # 77
    f.InitBoard(L, cells) # 100
    f.InitClusters(n, L) # 121

    t = f.RunSim() # 179
        # Verlet() - 207
        # VelHalfStep() - 220
        # Accel() - 240

        # SetKList() - 247

        # KinEnergy() - 264

    print(t)

    f.plot() # 287


if __name__ == '__main__':
    setup()
