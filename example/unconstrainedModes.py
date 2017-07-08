import sys
sys.path.append("pypace")
import constrainedPower as cnstpow
import numpy as np
from matplotlib import pyplot as plt

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
    plotSlices(mask)
    plotSlices(support)
    plt.show()

    constrained = cnstpow.ConstrainedPower( mask, support, Nbasis=7 )
    #constrained.checkOrthogonality()
    constrained.solve()
    constrained.plotEigenvalues()
    plt.show()

if __name__ == "__main__":
    main()
