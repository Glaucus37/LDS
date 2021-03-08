import Langevin as lan

def setup(L=10., n=10):
    lan.create_clusters(n)


if __name__ == '__main__':
    setup(n=10)
