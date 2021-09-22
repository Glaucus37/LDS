import Board as b
import time

if __name__ == '__main__':
    board = b.Board(t=1e3, dt=1e-2, l=2e6)
    board.RunSim()
