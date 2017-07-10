import sys
sys.path.append("pypace")
import constrainedPower as cnstpow
import numpy as np
from matplotlib import pyplot as plt
import pickle as pck

def plotSlices( data ):
    assert( len(data.shape) == 3 )
    center = int(data.shape[0]/2)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow( data[center,:,:], cmap="bone" )
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow( data[:,center,:], cmap="bone" )
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow( data[:,:,center], cmap="bone")
    return fig

def main():
    compareDenseSparse = False
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"
    scattered = np.load( kspace )
    realsp = np.load( reconstruct )
    realspPadded = np.zeros(scattered.shape)
    start = int( scattered.shape[0]/4 )
    end = int( 3*scattered.shape[0]/4 )

    scattered = scattered[::16,::16,::16]
    realspPadded[start:end,start:end,start:end] = realsp

    realsp = realspPadded[::16,::16,::16]
    mask = np.zeros( scattered.shape, dtype=np.uint8 )
    mask[scattered>1E-6*scattered.max()] = 1
    support = np.zeros( realsp.shape, dtype=np.uint8 )
    support[realsp>1E-6*realsp.max()] = 1

    constrained = cnstpow.ConstrainedPower( mask, support, Nbasis=8 )
    plotSlices(constrained.mask)
    plotSlices(constrained.support)
    plt.show()

    #constrained.checkOrthogonality()
    if ( compareDenseSparse ):
        eigval,eigvec = constrained.solve( mode="sparse", bandwidth=6, plotMatrix=True )
        eigvalD,eigvecD = constrained.solve( mode="dense", plotMatrix=True )
        plt.plot(eigval)
        plt.plot(eigvalD)
        plt.show()
    else:
        eigval,eigvec = constrained.solve( mode="dense", bandwidth=2, plotMatrix=False, fracEigmodes=0.1 )
    constrained.plotEigenvalues()
    plt.show()

    # Dump the results to a pickle file
    fname = "data/uncsontrainedModes.pck"
    out = open( fname, 'wb' )
    pck.dump( constrained, out )
    out.close()
    print ("Object result written to %s"%(fname))


if __name__ == "__main__":
    main()
