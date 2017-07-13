import sys
sys.path.append("pypace")
import unconstrainedModesCorrector as umc
import numpy as np
from matplotlib import pyplot as plt
import pickle as pck
import eDensityVisualizer as edv
from mayavi import mlab

def main():
    try:
        data = "data/average_NiAu_sample1_3D_50_1.npy"
        realspace = np.load( data )
        fname = "data/uncsontrainedModes.pck"
        infile = open(fname,'rb')
        cnstpow = pck.load(infile)
        infile.close()
    except Exception as exc:
        print (str(exc))
        return

    visualizer = edv.EDensityVisualizer()
    corrector = umc.UnconstrainedModeCorrector( cnstpow, realspace, minimizer="laplacian" )
    angles = [40,80,120,160]
    visualizer.plot1DAngles( angles, np.abs(corrector.data) )
    corrector.correct( 1 )
    visualizer.plot1DAngles( angles, np.abs(corrector.data) )
    visualizer.plotBest( np.abs( corrector.data[::8,::8,::8] ) )
    mlab.show()
    plt.show()

if __name__ == "__main__":
    main()
