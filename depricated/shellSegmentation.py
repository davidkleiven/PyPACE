from __future__ import print_function
import sys
sys.path.append("pypace")
from scipy import optimize as opt
from mayavi import mlab
from mpi4py import MPI
import azmDensityCorrector as adc

def main( argv ):
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"
    comm = MPI.COMM_WORLD
    dCorr = adc.SliceDensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm, debug=False,
    projectionAxis=2, segmentation="shell" )
    dCorr.segment( 6 )
    for i in range(6):
        dCorr.segmentor.plotCluster( i )
    mlab.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
