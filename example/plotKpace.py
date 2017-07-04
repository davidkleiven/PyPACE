import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def main():
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"

    ks = np.load( kspace )
    realsp = np.load(reconstruct)
    realspFull = np.zeros(ks.shape)
    start = int( ks.shape[0]/4 )
    end = int( 3*ks.shape[0]/4 )
    realspFull[start:end,start:end,start:end] = realsp
    kssim = np.abs( np.fft.fftn(realspFull) )**2
    kssim = np.fft.fftshift( kssim )

    center = int( ks.shape[0]/2 )

    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow( kssim[center,:,:], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow( kssim[:,center,:], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow( kssim[:,:,center], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow( ks[center,:,:], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow( ks[:,center,:], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    ax6 = fig.add_subplot(2,3,6)
    ax6.imshow( ks[:,:,center], cmap="nipy_spectral", interpolation="none", norm=mpl.colors.LogNorm() )
    plt.show()

if __name__ == "__main__":
    main()
