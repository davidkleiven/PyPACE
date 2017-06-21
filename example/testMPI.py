from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if ( rank == 0 ):
        array = np.linspace(0.0,10.0,20)
    else:
        array = np.linspace(10.0,20.0,20)
    dest = np.zeros(len(array))
    comm.Reduce(array,dest,op=MPI.SUM, root=0)
    if ( rank == 0 ):
        print ( dest )

if __name__ == "__main__":
    main()
