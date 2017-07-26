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
    """
    Analyse and can identify structures that only scatters into the region
    of missing data
    """
    def __init__( self, mask, support ):
        """
        MissingDataAnalyzer

        mask:
            3D array of type np.uint8. It is 1 if the voxel is measured and 0 if the voxel is not measured
        support:
            3D array of type np.uint8. It is 1 if the voxel belongs to the scatterer and zero if it is outside
        """
        self.mask = mask
        self.support = support
        self.beta = -0.9

        self.constraints = None
        self.fixedPoint = None
        self.allErrors = []
        self.bestImg = None
        self.bestError = 1E30

    def step( self, current ):
        """
        Iterate the solution one step

        current:
            List of the duplicates of the current image
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
        """
        Returns the fA function from https://en.wikipedia.org/wiki/Difference-map_algorithm

        x:
            List of the duplicates of the object
        """
        result = []
        for entry in x:
            result.append( copy.deepcopy(entry) )
        pa = self.projectA(x)
        for i in range(len(x)):
            x[i] = pa[i] - pa[i]/self.beta + result[i]/self.beta
        return x

    def fB( self, x ):
        """
        Returns the fB function in https://en.wikipedia.org/wiki/Difference-map_algorithm

        x:
            List of the duplicates of the object
        """
        pb = self.projectB( x )
        for i in range( len(x) ):
            x[i] = pb + pb/self.beta - x[i]/self.beta
        return x

    def projectB( self, x ):
        """
        Average all the duplicates of the object

        x:
            List of all the duplicates of the object
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
        Apply each the constraints to the corresponding duplicate

        x:
            List of all the duplicates of the object
        """
        #mdc.applyRealSpace( self, x )
        assert( len(x) == len(self.constraints) )
        for i in range(len(x)):
            x[i] = self.constraints[i].apply( x[i] )
        return x

    def plot( self, x ):
        """
        Plots slices in both real space and Fourier space of the image

        x:
            One representation of the object (note not a list of duplicates as in many of the other functions)
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

        img:
            The 3D object
        """
        norm1 = np.sqrt( np.sum(img**2) )
        outside = np.sqrt( np.sum( img[self.support==0]**2) )
        ft = np.abs( np.fft.fftn(img) )
        norm2 = np.sqrt( np.sum(ft**2) )
        inside = np.sqrt( np.sum(ft[self.mask==1]**2) )
        return 0.5*(outside/norm1 + inside/norm2)

    def getMaxVal( self, allimgs ):
        """
        Computes the maximum value in of all the duplicates

        allimgs:
            List of all the duplicates
        """
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
        Find a function that satisfy all the constraints given in the constraints list

        constraints: list
            List of constraints that the solution should satisfy

        nIter: int
            Maximum number of iterations

        relerror: float
            If the relative error between the objects produced when applying the different constraints is less
            than this number, the solution is found

        show: bool
            If True the plot of the resulting image is shown when the solution has been found or the maximum
            number of iterations have been reached

        initial: ndarray
            Initial condition from which the difference map starts from.
            Default is None, and then the initial object is filled with random numbers

        zeroLimit: float
            If the maximum value in the image is less than this number, the algorithm stops.
            The image is taken to consist of only zeros.
            An object consisting of only zeros will in many cases satisfy all the constraint and it is a trivial
            solution which is normally is not of interest.
            Hence, of the algorithm approaces this solution it can be of interest to just stop and pass another
            initial condition.

        Returns: dictionary
            Result dictionay having the keys
            message - A message describing the reason for terminating\n
            constrainedPower - the constrained power of the solution\n
            error - array of the relative error on each iteration step\n
            bestError - the minimum relative error\n
            status - True if the convergence criteria was reached, and False otherwise
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
    """
    Generic constraint
    """
    def __init__( self, missingData, weight=1.0 ):
        self.analyzer = missingData
        if ( not isinstance(missingData, MissingDataAnalyzer ) ):
            raise TypeError("missingData has to be of type MissingDataAnalyzer")
        self.weight = weight

    def apply( self, img ):
        """
        Applies the constraint to img. All child classes has to implement this function.

        img: ndarray
            An array representing the object.
        """
        raise NotImplementedError("Child classes has to implement this function")

class RealSpaceConstraint( Constraint ):
    """
    Realspace constraint.
    Voxels outside the support are set to zero and voxels inside the support
    are left unchaged.
    """
    def __init__( self, missingData, weight=1.0 ):
        """
        RealSpaceConstraint

        missingData: MissingDataAnalyzer
            An instance of the missingDataAnalyzer of which this constraint is passe to
        """
        Constraint.__init__( self, missingData, weight=weight )

    def apply( self, img ):
        """
        Applies the realspace constraint. Voxels inside the support of the missingDataAnalyzer are left unchanged,
        and voxels outisde are set to zero.

        img: ndarray
            The object on which the constraint is applied

        Returns: ndarray
            The object after the constraint have been applied
        """
        mdc.applyRealSpace( self.analyzer, img )
        return img

class FourierConstraint( Constraint ):
    """
    Fourier constraint. Intensity belonging to the region not being measured is left unchanged, and
    intensitis in the region of measured data is set to zero.
    """
    def __init__( self, missingData, weight=1.0 ):
        """
        FourierConstraint

        missingData: MissingDataAnalyzer
            Instance of the missingDataAnaluzer of which the constraint is passed
        """
        Constraint.__init__( self, missingData, weight=weight )
        self.ftsource = pfw.empty_aligned( missingData.mask.shape, dtype="complex128" )
        self.ftdest = pfw.empty_aligned( missingData.mask.shape, dtype="complex128" )
        self.ftForw = pfw.FFTW( self.ftsource, self.ftdest, axes=(0,1,2), threads=mp.cpu_count() )
        self.ftBack = pfw.FFTW( self.ftdest, self.ftsource, axes=(0,1,2), direction="FFTW_BACKWARD", threads=mp.cpu_count() )
        self.ftsource[:,:,:] = missingData.support

    def apply( self, img ):
        """
        Applies the Fourier constraint.
        Intensities inside the region of measured data is set to zero, and intensities outside this region
        is left unchanged

        img: ndarray
            The object of which the constraint is applied

        Returns: ndarray
            The real part of the inverse Fourier transform after the constraint in Fourier domain has been applied
        """
        self.ftsource[:,:,:] = img
        self.ftForw()
        ft = self.ftdest
        mdc.applyFourier( self.analyzer, ft )
        self.ftBack()
        return self.ftsource.real

class OrthogonalConstraint( Constraint ):
    """
    Orthogonality constraint.
    The image is forced to be orthogonal to this image
    """
    def __init__( self, missingData, orthogData, weight=1.0 ):
        """
        OrthogonalConstraint.

        missingData: MissingDataAnalyzer
            Instance of the missingDataAnalyzer of which this constraint is passed

        orthogData: ndarray
            Array representing the object of which the solution should be orthogonal to
        """
        Constraint.__init__( self, missingData, weight=weight )
        self.orthogData = orthogData
        self.orthogData /= np.sqrt( np.sum(self.orthogData**2) )

    def apply( self, img ):
        """
        Applies the ortogonal constraint.
        The object is made orthogonal to the data by subtracting the projecting of the current solution
        to the data array passed to this class.

        img: ndarray
            Array of the current object

        Returns: ndarray
            The new object after it has been made orthogonal to the image passed to this object
        """
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
