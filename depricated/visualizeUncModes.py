import sys
sys.path.append("pypace")
import unconstrainedModeVisualizer as umv
import pickle as pck
from mayavi import mlab
from matplotlib import pyplot as plt

def main():
    fname = "data/uncsontrainedModes.pck"
    try:
        infile = open(fname,'rb')
        cnstpow = pck.load( infile )
        infile.close()
    except Exception as exc:
        print (str(exc))
        return

    print (cnstpow.eigval)
    vis = umv.UnconstrainedModeVisualizer( cnstpow )
    vis.saveMaskToHDF5()
    #vis.plotSupport()
    vis.plotMask()
    vis.plotMode2DReal(0, maxVoxels=256)
    mlab.show()
    plt.show()

if __name__ == "__main__":
    main()
