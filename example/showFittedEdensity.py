import sys
sys.path.append("pypace")
import numpy as np
import eDensityVisualizer as edv
from matplotlib import pyplot as plt
from mayavi import mlab

def main():
    fname = "fittedElectronDensity.h5"

    visualzier = edv.EDensityVisualizer( fname )
    visualzier.plotBest()
    visualzier.plotBestRadialAveragedDensity()
    visualzier.plot1DAngles([0,40,80,120,160])
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    rec = np.load(reconstruct)
    visualzier.plot1DAngles( [0,40,80,120,160], data=rec )
    mlab.show()
    plt.show()

if __name__ == "__main__":
    main()
