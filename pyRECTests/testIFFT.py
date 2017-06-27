import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import ndimage

def main():
    data = np.load("pyREC/kspaceCoatedSphere3D.npy")
    com = ndimage.center_of_mass( data )
    print (com)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow( data[int(com[0]),:,:], norm=mpl.colors.LogNorm())
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow( data[:,int(com[0]),:], norm=mpl.colors.LogNorm())
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow( data[:,:,int(com[2])], norm=mpl.colors.LogNorm())
    plt.show()

    invFFT = np.fft.ifftn( data )
    data = np.abs( np.fft.fftshift( invFFT ) )
    #data = np.abs(invFFT)

    com = ndimage.center_of_mass( data )
    print (com)

    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow( data[int(com[0]),:,:], norm=mpl.colors.LogNorm())
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow( data[:,int(com[0]),:], norm=mpl.colors.LogNorm())
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow( data[:,:,int(com[2])], norm=mpl.colors.LogNorm())
    plt.show()

if __name__ == "__main__":
    main()
