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
    #visualzier.plotBestRadialAveragedDensity()
    visualzier.plotCluster( 5 )
    mlab.show()
    plt.show()

if __name__ == "__main__":
    main()
