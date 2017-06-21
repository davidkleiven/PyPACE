from __future__ import print_function
import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import densityCorrector as dc
from matplotlib import pyplot as plt
from mpi4py import MPI

def main( argv ):
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"

    if ( len(argv) != 1 ):
        print ("Usage: python ugelStadSphere.py [--exploreData,--fit]")
        return
    comm = MPI.COMM_WORLD
    dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm )
    for arg in argv:
        if ( arg.find("--exploreData") != -1 ):
            dCorr.segment( 6 )
            dCorr.segmentor.replaceDataWithMeans()
            dCorr.plotKspace( dCorr.kspace )
            dCorr.buildKspace( 10.0 )
            dCorr.plotKspace( dCorr.newKspace )
            plt.show()
        elif ( arg.find("--fit") != -1 ):
            dCorr.fit( 6 )

    comm.Disconnect()

if __name__ == "__main__":
    main( sys.argv[1:] )
