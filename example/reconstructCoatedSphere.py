import sys
sys.path.append("pyREC")
import reconstructor as rec
import objectToScatteredTransformer as otst
import numpy as np
from matplotlib import pyplot as plt

def main():
    fname = "pyREC/kspaceCoatedSphere3D.npy"
    kspace = np.load(fname)
    rytov = otst.Rytov( kspace, 1.0 )
    born = otst.FirstBorn( kspace )
    reconstructor = rec.Reconstructor( born, 0.05, maxIter=200 )
    reconstructor.run()
    reconstructor.plotResidual()
    reconstructor.plotCurrent()
    reconstructor.plotBest()
    plt.show()

if __name__ == "__main__":
    main()
