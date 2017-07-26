from __future__ import print_function
import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import densityCorrector as dc
from matplotlib import pyplot as plt
from mpi4py import MPI

reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
kspace = "data/NiAu_sample1_3D.npy"

def main( argv ):

    if ( len(argv) != 1 ):
        print ("Usage: python ugelStadSphere.py [--exploreData,--fit,--testFit]")
        return
    comm = MPI.COMM_WORLD
    dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm, debug=False )
    for arg in argv:
        if ( arg.find("--exploreData") != -1 ):
            dCorr.segment( 10 )
            dCorr.segmentor.replaceDataWithMeans()
            dCorr.plotKspace( dCorr.kspace )
            dCorr.buildKspace( 10.0 )
            dCorr.plotKspace( dCorr.newKspace )
            dCorr.plotKspace( dCorr.kspace )
            plt.show()
        elif ( arg.find("--fit") != -1 ):
            dCorr.fit( 10, angleStepKspace=20, nGAgenerations=1000 )
        elif ( arg.find("--testFit") != -1 ):
            dCorr.fit( 2, nGAgenerations=1 )

if __name__ == "__main__":
    main( sys.argv[1:] )
