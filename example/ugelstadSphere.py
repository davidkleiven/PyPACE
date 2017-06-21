from __future__ import print_function
import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import densityCorrector as dc
from matplotlib import pyplot as plt
from mpi4py import MPI

def main():
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"

    comm = MPI.COMM_WORLD
    dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm )
    dCorr.segment( 6 )
    comm.disconnect()
    dCorr.plotRec()
    #dCorr.segmentor.replaceDataWithMeans()
    #dCorr.plotClusters(0)
    #dCorr.plotClusters(1)
    #dCorr.plotClusters(2)
    #dCorr.plotClusters(3)
    #dCorr.plotKspace( dCorr.kspace )
    #dCorr.buildKspace( 10.0 )
    #dCorr.plotKspace( dCorr.newKspace )
    #dCorr.qweight.compute(showPlot=True)
    plt.show()

if __name__ == "__main__":
    main()
