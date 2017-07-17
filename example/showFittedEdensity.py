import sys
sys.path.append("pypace")
import numpy as np
import eDensityVisualizer as edv
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from mayavi import mlab

def main():
    fname = "fittedElectronDensity.h5"

    visualzier = edv.EDensityVisualizer( fname )
    visualzier.plotBest()
    mlab.show()
    visualzier.plotBestRadialAveragedDensity()
    visualzier.plot1DAngles([0,40,80,120,160])
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    rec = np.load(reconstruct)
    visualzier.plot1DAngles( [0,40,80,120,160], data=rec )
    visualzier.plotFit()
    visualzier.plotBest( data=rec.astype(np.float64) )
    mlab.show()

    visualzier.plotOutline( data=rec.astype(np.float64)[::2,::2,::2] )
    mlab.show()
    plt.show()

if __name__ == "__main__":
    main()
