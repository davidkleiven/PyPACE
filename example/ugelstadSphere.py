import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import densityCorrector as dc
from matplotlib import pyplot as plt

def main():
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"

    dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2 )
    dCorr.plotRec()
    #dCorr.segment( 6 )
    #dCorr.segmentor.replaceDataWithMeans()
    #dCorr.plotClusters(0)
    #dCorr.plotClusters(1)
    #dCorr.plotClusters(2)
    #dCorr.plotClusters(3)
    dCorr.plotKspace( dCorr.kspace )
    dCorr.buildKspace( 10.0 )
    dCorr.plotKspace( dCorr.newKspace )
    dCorr.qweight.compute(showPlot=True)
    plt.show()

if __name__ == "__main__":
    main()
