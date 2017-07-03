from __future__ import print_function
import numpy as np
import matplotlib as mpl
import config
if ( not config.enableMPLShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt

def main():
    use3DFT = True
    au = 8.5E-6
    pmma = 8.5E-6
    N =128
    delta = np.zeros((N,N,N))
    shape = delta.shape
    Rau = 10
    Rpmma = 10
    del delta
    x = np.linspace(-N/2,N/2,shape[0])
    y = np.linspace(-N/2,N/2,shape[1])
    z = np.linspace(-N/2,N/2,shape[2])
    X,Y,Z = np.meshgrid(x,y,z)
    R = np.sqrt( X**2 + Y**2 + Z**2 )
    del X,Y,Z
    delta = np.zeros((N,N,N))
    delta[R < Rau ] = au
    delta[ R < Rpmma ] = pmma
    dx = 2.0 # Voxel size in nm

    # Compute far field using projection
    wavelength = 0.17
    k = 2.0*np.pi/wavelength
    proj = k*delta.sum(axis=2)
    if ( not use3DFT ):
        del delta
    del R
    ff = np.fft.fft2( np.exp(1j*proj*dx)-1.0 )/N
    ff = np.abs( np.fft.fftshift(ff) )**2

    plt.figure(1)
    plt.imshow(ff, norm=mpl.colors.LogNorm(), interpolation="none", cmap="inferno")
    plt.figure(2)
    plt.imshow(proj, interpolation="none", cmap="inferno")
    plt.colorbar()
    plt.show()
    ff = ff[:,N/2]

    del proj

    if ( use3DFT ):
        kspace = np.abs( np.fft.fftn( delta, norm="ortho" ) )
        kspace = np.fft.fftshift(kspace)
    else:
        # Create the 3D far field pattern
        kspace = np.zeros((N,N,N))
        print ("Fill 3D matrix", end="\r")
        minval = 1E7
        for i in range(0,N):
            print ("%d of %d"%(i,N))
            kx = i-N/2
            for j in range(0,N):
                ky = j-N/2
                for k in range(0,N):
                    kz = k-N/2
                    kr = np.sqrt(kx**2 + ky**2+kz**2)
                    if ( kr < N/2 ):
                        if ( int(kr) < N/2-1 ):
                            weight = kr-int(kr)
                            kspace[i,j,k] = ff[N/2+int(kr)+1]*weight + (1.0-weight)*ff[N/2+int(kr)]
                        else:
                            kspace[i,j,k] = ff[N/2+int(kr)]
                        if ( kspace[i,j,k] < minval ):
                            minval = kspace[i,j,k]
        kspace[kspace<minval] = minval
    plt.figure(3)
    plt.imshow(kspace[N/2,:,:], norm=mpl.colors.LogNorm(), interpolation="none", cmap="inferno")
    plt.figure(4)
    plt.imshow(kspace[:,N/2,:], norm=mpl.colors.LogNorm(), interpolation="none", cmap="inferno")
    plt.figure(5)
    plt.imshow(kspace[:,:,N/2], norm=mpl.colors.LogNorm(), interpolation="none", cmap="inferno")
    plt.show()
    fname = "kspaceCoatedSphere3D.npy"
    np.save(fname, kspace)
    print ("3D Scattered array saved to %s"%(fname))

if __name__ == "__main__":
    main()
