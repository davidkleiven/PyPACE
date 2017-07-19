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
        self.beta = -0.9

        #self.ftsource = pfw.empty_aligned( self.mask.shape, dtype="complex128" )
        #self.ftdest = pfw.empty_aligned( self.mask.shape, dtype="complex128" )
        #self.ftForw = pfw.FFTW( self.ftsource, self.ftdest, axes=(0,1,2), threads=mp.cpu_count() )
        #self.ftBack = pfw.FFTW( self.ftdest, self.ftsource, axes=(0,1,2), direction="FFTW_BACKWARD", threads=mp.cpu_count() )
        #self.ftsource[:,:,:] = self.support
        self.constraints = None

    def step( self, current ):
        x = copy.deepcopy( current )
        pa = self.projectA( self.fB(x) )
        #assert( np.allclose(x,current) )
        x = copy.deepcopy( current )
        pb = self.projectB( self.fA(x) )

        #error = np.abs( pa-pb ).max()
        #print ("Error: %.2E"%(error))
        #current = current + self.beta*( pa-pb )
        #error = np.abs( )
        error = []
        maxvals = []
        for i in range(len(current)):
            error.append( np.abs(pa[i]-pb).max()/np.abs(pb.max()) )
            current[i] = current[i] + self.beta*( pa[i]-pb )
            maxvals.append( current[i].max() )
        print ( "Error: %.2E"%(np.array(error).max()) )
        return current, np.array(error).max()

    def fA( self, x ):
        #result = x/self.beta
        #pa = self.projectA(x) # This modifies x
        #return pa - pa/self.beta + result
        result = []
        for entry in x:
            result.append( copy.deepcopy(entry) )
        pa = self.projectA(x)
        for i in range(len(x)):
            x[i] = pa[i] - pa[i]/self.beta + result[i]/self.beta
        return x

    def fB( self, x ):
        #res = x/self.beta
        #pb = self.projectB(x)
        #return pb + pb/self.beta - res
        pb = self.projectB( x )
        for i in range( len(x) ):
            x[i] = pb + pb/self.beta - x[i]/self.beta
        return x

    def projectB( self, x ):
        """
        Make x-consistent with the Fourier domain constraint
        """
        #self.ftsource[:,:,:] = x
        #self.ftForw()
        #ft = self.ftdest
        #mdc.applyFourier( self, ft )
        #self.ftBack()
        avg = np.zeros(x[0].shape)
        # Normalize
        for i in range(len(x)):
            avg += x[i]
        return avg/len(x)
        #return self.ftsource.real

    def projectA( self, x ):
        """
        Make x-consistent with the real space constraint
        """
        #mdc.applyRealSpace( self, x )
        assert( len(x) == len(self.constraints) )
        for i in range(len(x)):
            x[i] = self.constraints[i].apply( x[i] )
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
        ax4.imshow( msk[center,:,:], cmap="bone", alpha=0.3)

        ax5 = fig.add_subplot(2,3,5)
        ax5.imshow( ft[:,center,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm())
        ax5.imshow( msk[:,center,:], cmap="bone", alpha=0.3)

        ax6 = fig.add_subplot(2,3,6)
        ax6.imshow( ft[:,:,center], cmap="nipy_spectral", norm=mpl.colors.LogNorm())
        ax6.imshow( msk[:,:,center], cmap="bone", alpha=0.3)
        return fig

    def computeConstrainedPower( self, img ):
        norm1 = np.sqrt( np.sum(img**2) )
        outside = np.sqrt( np.sum( img[self.support==0]**2) )
        ft = np.abs( np.fft.fftn(img) )
        norm2 = np.sqrt( np.sum(ft**2) )
        inside = np.sqrt( np.sum(ft[self.mask==1]**2) )
        return 0.5*(outside/norm1 + inside/norm2)

    def getImg( self, current ):
        pa = self.projectA(current)
        return self.projectB(pa)

    def solve( self, constraints, niter=1000, relerror=0.0 ):
        self.constraints = constraints
        current = [copy.deepcopy(self.support).astype(np.float64) for i in range(len(constraints))]
        #current = self.support
        self.mask = np.fft.fftshift(self.mask)
        for i in range(niter):
            print ("Iteration %d of maximum %d"%(i,niter))
            current, error = self.step( current )
            if ( error < relerror):
                print ("Convergence criteria reached")
                break


        # Now the projection of the image to constraint A also satisfies the constrained B
        current = self.getImg( current )
        fname = "data/unconstrained.dat"
        np.save( fname, current )
        print ("Constrained power %.2E"%(self.computeConstrainedPower(current)))
        print ("Data written to %s"%(fname))
        self.plot( current )
        plt.show()

# Define constraint classes
class Constraint( object ):
    def __init__( self, missingData ):
        self.analyzer = missingData
        if ( not isinstance(missingData, MissingDataAnalyzer ) ):
            raise TypeError("missingData has to be of type MissingDataAnalyzer")

    def apply( self, img ):
        raise NotImplementedError("Child classes has to implement this function")

class RealSpaceConstraint( Constraint ):
    def __init__( self, missingData ):
        Constraint.__init__( self, missingData )

    def apply( self, img ):
        mdc.applyRealSpace( self.analyzer, img )
        return img

class FourierConstraint( Constraint ):
    def __init__( self, missingData ):
        Constraint.__init__( self, missingData )
        self.ftsource = pfw.empty_aligned( missingData.mask.shape, dtype="complex128" )
        self.ftdest = pfw.empty_aligned( missingData.mask.shape, dtype="complex128" )
        self.ftForw = pfw.FFTW( self.ftsource, self.ftdest, axes=(0,1,2), threads=mp.cpu_count() )
        self.ftBack = pfw.FFTW( self.ftdest, self.ftsource, axes=(0,1,2), direction="FFTW_BACKWARD", threads=mp.cpu_count() )
        self.ftsource[:,:,:] = missingData.support

    def apply( self, img ):
        self.ftsource[:,:,:] = img
        self.ftForw()
        ft = self.ftdest
        mdc.applyFourier( self.analyzer, ft )
        self.ftBack()
        return self.ftsource.real
