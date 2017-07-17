from __future__ import print_function
import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import azmDensityCorrector as adc
from matplotlib import pyplot as plt
from mpi4py import MPI
from mayavi import mlab

def main( argv ):
    nIter = 12
    for arg in argv:
        if ( arg.find("--iter=") != -1 ):
            nIter = int( arg.split("--iter=")[1] )

    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"
    comm = MPI.COMM_WORLD
    dCorr = adc.SliceDensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm, debug=False,
    projectionAxis=2, segmentation="voxels" )
    dCorr.segment( 6 )
    #dCorr.segmentor.replaceDataWithMeans()
    dCorr.segmentor.projectClusters()
    dCorr.plotSliceKspace()
    dCorr.plotMask()
    dCorr.segmentor.plotCluster(3, downsample=4)

    # Try to fill the missing data in the Fourier domain by a gaussian approximation
    #dCorr.qweight.fillMissingDataWithGaussian( dCorr.mask )

    #dCorr.saveAllSliceClusters()
    print ("Optimizing parameters")
    width = int( dCorr.kspace.shape[0]/16 )
    dCorr.fit( nIter=nIter, nClusters=6, maxDelta=1E-4, useSeparateClusterAtCenter=True, centerClusterWidth=width )
    dCorr.merge()
    dCorr.plotFit( optimum["x"] )
    #plt.show()
    #mlab.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
