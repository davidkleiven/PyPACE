import sys
sys.path.append("pypace")
import numpy as np
import removeUncoveredRegion as rur

def main( argv ):
    fname = "data/Run5pFull/unconstrainedModes2017_07_22_11_33_40.h5"
    realsp = np.load( "data/average_NiAu_sample1_3D_50_1.npy" )
    remover = rur.RemoveUncovered( realsp, fname )
    removed = remover.projectToScattered( asint8=True )
    fnameOut = "data/realspaceCorrected.npy"
    np.save( fnameOut, removed )
    print ( "Projected version saved to %s"%(fnameOut) )
    if ( "--plot" in argv ):
        remover.plot()

if __name__ == "__main__":
    main( sys.argv[1:] )
