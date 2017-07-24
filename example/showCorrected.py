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
    recs = np.load( "data/realspaceCorrected.npy" )
    visualizer = edv.EDensityVisualizer()
    #visualizer.plotBest( data=recs.astype(np.float16)[::4,::4,::4] )
    #mlab.show()
    fig = visualizer.plot1DAngles( [0,40,80,120,160], data=recs )
    plt.show()

if __name__ == "__main__":
    main()
