import sys
sys.path.append("pypace")
import unconstrainedModesCorrector as umc
import numpy as np
from matplotlib import pyplot as plt
import pickle as pck
import eDensityVisualizer as edv

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
    corrector = umc.UnconstrainedModeCorrector( cnstpow, realspace, minimizer="gradient" )
    angles = [40,80,120,160]
    visualizer.plot1DAngles( angles, np.abs(corrector.data) )
    corrector.correct( 1E-15 )
    visualizer.plot1DAngles( angles, np.abs(corrector.data) )
    plt.show()

if __name__ == "__main__":
    main()
