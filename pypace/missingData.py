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
import subprocess as sub

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
        self.fixedPoint = None
        self.allErrors = []
        self.bestImg = None
        self.bestError = 1E30

    def step( self, current ):
        """
        Iterate the solution one step
        """
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
        totweight = 0
        for i in range(len(x)):
            avg += x[i]*self.constraints[i].weight
            totweight += self.constraints[i].weight
        return avg/totweight
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
        """
        Plots slices in both real space and Fourier space of the image
        """
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

    def plotProgression( self ):
        """
        Plots the error on each iteration step
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(self.allErrors, color="black")
        ax1.set_yscale("log")
        return fig

    def computeConstrainedPower( self, img ):
        """
        Computes the fraction of constrained power by the realspace constraint and the Fourier constraint
        """
        norm1 = np.sqrt( np.sum(img**2) )
        outside = np.sqrt( np.sum( img[self.support==0]**2) )
        ft = np.abs( np.fft.fftn(img) )
        norm2 = np.sqrt( np.sum(ft**2) )
        inside = np.sqrt( np.sum(ft[self.mask==1]**2) )
        return 0.5*(outside/norm1 + inside/norm2)

    def getMaxVal( self, allimgs ):
        maxval = -1E30
        for img in allimgs:
            trial = np.abs(img).max()
            if ( trial > maxval ):
                maxval = trial
        return maxval

    def getImg( self ):
        """
        Computes the resulting image
        """
        if ( self.fixedPoint is None ):
            raise RuntimeError("Cannot produce image when the solve function is not called")
        pa = self.projectA( self.fixedPoint )
        return self.projectB(pa)

    def solve( self, constraints, niter=1000, relerror=0.0, show=False, initial=None, zeroLimit=0.0 ):
        """
        Find a function that satisfy the real space constraint, but scatter only into the region that is not measured
        Arguments:
            niter - maximum number of iterations
            relerror - convergence criteria. The simulation stops when max(Pa-Pb)/max(Pb) < relerror
                Pa is the result of the first projection operator and Pb is the result of the second
            show - Boolean if True a plot of the solution is showed
            initial - initial image. If None, the image is initialized with random numbers
            zeroLimit - Upper limit of what is considered as the zero image. If the maximum value
                of the image is below this number, the image is considered to consist of only zeros and the simulation is
                aborted. Note that the zero image typically satisfy all the constraints perfectly, so one will never escape
                from the zero image.
        """
        self.constraints = constraints
        for cnst in self.constraints:
            if ( not isinstance(cnst,Constraint) ):
                raise TypeError("All constraints need to be of type Constraint")

        shp = self.support.shape
        if ( initial is None or initial.shape != self.mask.shape ):
            print ("Generating initial state...")
            initial = np.random.rand(shp[0],shp[1],shp[2])
        current = [copy.deepcopy(initial) for i in range(len(constraints))]
        #current = self.support
        self.mask = np.fft.fftshift(self.mask)
        sub.call(["mkdir","-p","localMinima"])
        saveImgNextTimeErrorIncrease = True
        message = ""
        status = False
        for i in range(niter):
            print ("Iteration %d of maximum %d"%(i,niter))
            current, error = self.step( current )
            self.allErrors.append(error)

            if ( error < self.bestError ):
                print ("New best image...")
                self.bestError = error
                self.fixedPoint = current
                self.bestImg = copy.deepcopy( self.getImg() )

            if ( error < relerror):
                message = "Convergence criteria reached"
                status = True
                break

            if ( error > self.allErrors[-1] and saveImgNextTimeErrorIncrease ):
                self.fixedPoint = current
                fig = self.plot( self.getImg() )
                fig.savefig("localMinima/img%d.png"%(i))
                saveImgNextTimeErrorIncrease = False
            else:
                saveImgNextTimeErrorIncrease = True

            if ( self.getMaxVal(current) < zeroLimit ):
                message = "Approaching the zero solution. Aborted."
                break

        # Now the projection of the image to constraint A also satisfies the constrained B
        self.fixedPoint = current
        current = self.getImg()
        fname = "data/unconstrained.dat"
        np.save( fname, current )
        cnstpower = self.computeConstrainedPower(current)
        print ("Constrained power %.2E"%(self.computeConstrainedPower(current)))
        print ("Data written to %s"%(fname))
        print (message)
        if ( show ):
            self.plot( self.bestImg )
            self.plotProgression()
            plt.show()

        res = {"message":message,
        "image":self.bestImg,
        "constrainedPower":cnstpower,
        "error":self.allErrors,
        "bestError":self.bestError,
        "status":status}
        return res

# Define constraint classes
class Constraint( object ):
    def __init__( self, missingData, weight=1.0 ):
        self.analyzer = missingData
        if ( not isinstance(missingData, MissingDataAnalyzer ) ):
            raise TypeError("missingData has to be of type MissingDataAnalyzer")
        self.weight = weight

    def apply( self, img ):
        raise NotImplementedError("Child classes has to implement this function")

class RealSpaceConstraint( Constraint ):
    def __init__( self, missingData, weight=1.0 ):
        Constraint.__init__( self, missingData, weight=weight )

    def apply( self, img ):
        mdc.applyRealSpace( self.analyzer, img )
        return img

class FourierConstraint( Constraint ):
    def __init__( self, missingData, weight=1.0 ):
        Constraint.__init__( self, missingData, weight=weight )
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

class OrthogonalConstraint( Constraint ):
    def __init__( self, missingData, orthogData, weight=1.0 ):
        Constraint.__init__( self, missingData, weight=weight )
        self.orthogData = orthogData
        self.orthogData /= np.sqrt( np.sum(self.orthogData**2) )

    def apply( self, img ):
        # Project onto
        proj = np.sum( img*self.orthogData )
        img -= proj*self.orthogData
        return img

class NormalizationConstraint( Constraint ):
    def __init__( self, missingData ):
        Constraint.__init__( self, missingData )

    def apply( self, img ):
        img /= np.sqrt( np.sum(img**2) )
        return img
