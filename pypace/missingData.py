import sys
sys.path.append("./")
sys.path.append("pypace")
sys.path.append("../")
import numpy as np
import missingdatac as mdc
import copy
import config
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
import pyfftw as pfw
import multiprocessing as mp

class MissingDataAnalyzer( object ):
    def __init__( self, mask, support ):
        self.mask = mask
        self.support = support
        self.beta = 0.9

        self.ftsource = pfw.empty_aligned( self.mask.shape, dtype="complex128" )
        self.ftdest = pfw.empty_aligned( self.mask.shape, dtype="complex128" )
        self.ftForw = pfw.FFTW( self.ftsource, self.ftdest, axes=(0,1,2), threads=mp.cpu_count() )
        self.ftBack = pfw.FFTW( self.ftdest, self.ftsource, axes=(0,1,2), direction="FFTW_BACKWARD", threads=mp.cpu_count() )
        self.ftsource[:,:,:] = self.support

    def step( self, current ):
        x = copy.deepcopy( current )
        pa = self.projectA( self.fB(x) )
        assert( np.allclose(x,current) )
        pb = self.projectB( self.fA(x) )

        error = np.abs( pa-pb ).max()
        print ("Error: %.2E"%(error))
        current = current + self.beta*( pa-pb )
        return current

    def fA( self, x ):
        result = x/self.beta
        pa = self.projectA(x) # This modifies x
        return pa - pa/self.beta + result

    def fB( self, x ):
        res = x/self.beta
        pb = self.projectB(x)
        return pb + pb/self.beta - res

    def projectB( self, x ):
        """
        Make x-consistent with the Fourier domain constraint
        """
        #ft = np.fft.fftn( x )
        self.ftsource[:,:,:] = x
        self.ftForw()
        ft = self.ftdest
        mdc.applyFourier( self, ft )
        self.ftBack()
        return self.ftsource.real
        return np.fft.ifftn( ft ).real

    def projectA( self, x ):
        """
        Make x-consistent with the real space constraint
        """
        mdc.applyRealSpace( self, x )
        return x

    def plot( self, x ):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        center = int( self.support.shape[0]/2 )
        im = ax1.imshow( x[center,:,:], cmap="inferno")
        fig.colorbar(im)
        ax1.imshow( self.support[center,:,:], cmap="bone", alpha=0.3 )

        ax2 = fig.add_subplot(2,3,2)
        im = ax2.imshow( x[:,center,:], cmap="inferno")
        fig.colorbar(im)
        ax2.imshow( self.support[:,center,:], cmap="bone", alpha=0.3)

        ax3 = fig.add_subplot(2,3,3)
        im = ax3.imshow( x[:,:,center], cmap="inferno")
        fig.colorbar(im)
        ax3.imshow( self.support[:,:,center], cmap="bone", alpha=0.3)

        msk = np.fft.fftshift(self.mask)
        ft = np.abs( np.fft.fftn(x) )
        ft = np.fft.fftshift(ft)
        ax4 = fig.add_subplot(2,3,4)
        ax4.imshow( ft[center,:,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm() )
        ax4.imshow( msk[center,:,:], cmap="bone", alpha=0.5)

        ax5 = fig.add_subplot(2,3,5)
        ax5.imshow( ft[:,center,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm())
        ax5.imshow( msk[:,center,:], cmap="bone", alpha=0.5)

        ax6 = fig.add_subplot(2,3,6)
        ax6.imshow( ft[:,:,center], cmap="nipy_spectral", norm=mpl.colors.LogNorm())
        ax6.imshow( msk[:,:,center], cmap="bone", alpha=0.5)
        return fig

    def computeConstrainedPower( self, img ):
        norm1 = np.sqrt( np.sum(img**2) )
        outside = np.sqrt( np.sum( img[self.support==0]**2) )
        ft = np.abs( np.fft.fftn(img) )
        norm2 = np.sqrt( np.sum(ft**2) )
        inside = np.sqrt( np.sum(ft[self.mask==1]**2) )
        return 0.5*(outside/norm1 + inside/norm2)

    def solve( self, niter=1000 ):
        current = self.support
        self.mask = np.fft.fftshift(self.mask)
        for i in range(niter):
            print ("Iteration %d of maximum %d"%(i,niter))
            current = self.step( current )

        # Now the projection of the image to constraint A also satisfies the constrained B
        currentB = self.projectB(current)
        currentA = self.projectA(current)
        current = 0.5*(currentB+currentA)
        fname = "data/unconstrained.dat"
        np.save( fname, current )
        print ("Constrained power %.2E"%(self.computeConstrainedPower(current)))
        print ("Data written to %s"%(fname))
        self.plot( current )
        plt.show()
