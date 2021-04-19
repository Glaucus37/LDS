import functions as f
import klist as k
import matplotlib.pyplot as plt

# Particle - 9
# Cluster - 28

def Setup(L=10, n=1, t=1e6, dt=1e-2):
    f.TimeSetup(t, dt) # 125
    f.InitBoard(L) # 148
    f.InitClusters(n, L) # 168
    f.SetNeighbours() # 219


def TimeGraph():
    Setup()
    t11 = f.RunSim()
    print(t11)

    Setup(100)
    t21 = f.RunSim()
    print(t21)

    """
    fig, ax = plt.subplots(1, 1)
    plt.plot([1, 2], t11, t21)
    plt.suptitle('Time vs. Sidelength')
    plt.xticks([1, 2], ['10', '100'])
    # plt.scatter([1, 2], [t11, t21])

    plt.show()
    """


if __name__ == '__main__':
    # k.DebugKList()

    # TimeGraph()

    Setup()

    t = f.RunSim() # 244
        # Verlet() - 257
        # VelHalfStep() - 276
        # Accel() - 298

        # CheckNeighbors() - 332
        # JoinClusters() - 333
        # Momentum() - 369

        # KinEnergy() - 378

        # PlotClusters() - 398

    print(t)

    f.plot() # 460
