import functions as f

def setup(t=1e2, dt=1e-2, L=10., cells=4, n=10):
    f.time_setup(t, dt)

    f.init_board(L, cells)
    f.init_clusters(n)

    f.run_sim()

    plots.plot_energy(energy)

if __name__ == '__main__':

    setup()
