import sys
sys.path.append("pyREC")
import reconstructor as rec
import objectToScatteredTransformer as otst
import numpy as np
import matplotlib as mpl
import config
if ( not config.enableMPLShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
import initialSupports as isup

def main( argv ):
    fname = argv[0]
    kspace = np.load(fname)
    rytov = otst.Rytov( kspace, 1.0 )
    born = otst.FirstBorn( kspace, numpyFFT=False )
    initSup = isup.SphericalSupport( kspace.shape[0], 10, 1E-4 )
    #initSup = isup.BoxSupport( kspace.shape[0], 40, 1E-4 )
    reconstructor = rec.Reconstructor( born, 0.05, beta=1.0, maxIter=200 )
    reconstructor.initDataWithKnownSupport( initSup )
    #reconstructor.initScatteredDataWithRandomPhase()
    reconstructor.run( graphicUpdate=False )
    reconstructor.plotResidual()
    figCurr = reconstructor.plotCurrent()
    figBest = reconstructor.plotBest()
    figCurr1D = reconstructor.plot1DCuts()
    figCurr.savefig("current.png")
    figBest.savefig("best.png")
    reconstructor.save()
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
