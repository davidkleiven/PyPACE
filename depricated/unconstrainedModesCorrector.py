import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndimg
from scipy import optimize as opt
import constrainedPower as cpow

class UnconstrainedModeCorrector( object ):
    def __init__( self, cnstpow, data, minimizer="laplacian" ):
        if ( isinstance(cnstpow, cpow.ConstrainedPower) or isinstance(cnstpow, cpow.ConstrainedPowerFromFile) ):
            self.cnstpow = cnstpow
        else:
            raise TypeError("cnstpow has to be either an instance of ConstrainedPower or ConstrainedPowerFromFile")
        self.data = data

        if ( minimizer == "laplacian"):
            self.minimizer = LaplacianSquared( self.data, self.cnstpow, ds=8 )
        elif ( minimizer == "gradient" ):
            self.minimizer = GradientSquared( self.data, self.cnstpow, ds=8 )
        elif ( minimizer == "variance" ):
            self.minimizer = Variance( self.data, self.cnstpow, ds=8 )
        else:
            raise ValueError("minimizer has to be laplacian, gradient or variance")

    def correct( self, nModes, useComplexCoeff=False ):
        if ( nModes == 0 ):
            raise RuntimeError("Number of modes has to be larger than zero")

        self.minimizer.nModes = nModes
        print ( "Using %d least constrained modes"%(nModes) )

        self.minimizer.preCalculateEigenmodes()
        if ( isinstance(self.cnstpow,cpow.ConstrainedPower) ):
            minmethod="Nelder-Mead"
        elif ( isinstance(self.cnstpow,cpow.ConstrainedPower) ):
            minmethod="BFGS"

        print ("Optimizing results")

        options = {"maxiter":10000,
                    "disp":True}
        if ( useComplexCoeff ):
            print ("Using complex coefficients...")
            x0 = np.ones(2*nModes)
        else:
            print ("Using real coefficients...")
            x0 = np.ones(nModes)

        res = opt.minimize( self.minimizer.evaluate, x0, method=minmethod, options=options )
        x = np.array( res["x"] )
        self.data = self.minimizer.subtractModes( x )
        print ( res["message"] )
        print ( "Number of iterations %d"%( res["nit"]) )


class MinimizationTarget( object ):
    def __init__( self, data, cnstpow, ds=1 ):
        self.data = data[::ds,::ds,::ds]
        self.ds = ds
        self.cnstpow = cnstpow
        self.sup = self.cnstpow.support[::2*ds,::2*ds,::2*ds]
        self.nModes = 0
        self.eigmodes = []

    def preCalculateEigenmodes( self ):
        if ( isinstance(self.cnstpow, cpow.ConstrainedPower) ):
            N = self.data.shape[0]
            x = np.linspace(-N*self.ds*self.cnstpow.voxelsize/2,N*self.ds*self.cnstpow.voxelsize/2,N)
            X,Y,Z = np.meshgrid( x, x, x )
            for i in range(self.nModes):
                neweigmode = self.cnstpow.getEigenModeReal( i, X,Y,Z )
                self.eigmodes.append(neweigmode)
        elif ( isinstance(self.cnstpow,cpow.ConstrainedPowerFromFile)):
            self.eigmodes.append(self.cnstpow.data)
        else:
            raise TypeError("cnstpow has to be of type ConstrainedPower or ConstrainedPowerFromFile")

    def subtractModes( self, coeff ):
        if ( self.eigmodes == [] ):
            raise ValueError("The function preCalculateEigenmodes needs to be called")

        corrected = np.zeros(self.data.shape, dtype=np.complex64)
        corrected[:,:,:] = self.data
        for i in range(self.nModes):
            corrected -= coeff[i]*self.eigmodes[i]
        return corrected

    def evaluate( self, coeff ):
        raise NotImplementedError( "Child classes has to implement this function" )

class LaplacianSquared( MinimizationTarget ):
    def __init__( self, data, cnstpow, ds=1 ):
        MinimizationTarget.__init__( self, data, cnstpow , ds=ds)

    def evaluate( self, coeff ):
        Ncoeff = int( len(coeff)/2 )
        coeff = coeff[:Ncoeff] + 1j*coeff[Ncoeff:]
        corrected = self.subtractModes( coeff )
        laplace = ndimg.filters.laplace( np.abs(corrected), mode="constant", cval=0.0 )
        return np.sum( laplace[self.sup==1]**2 )

class GradientSquared( MinimizationTarget ):
    def __init__( self, data, cnstpow, ds=1 ):
        MinimizationTarget.__init__( self, data, cnstpow, ds=ds)

    def evaluate( self, coeff ):
        Ncoeff = int( len(coeff)/2 )
        coeff = coeff[:Ncoeff]# + 1j*coeff[Ncoeff:]
        corrected = np.abs( self.subtractModes( coeff ) )
        grad = np.gradient( corrected )
        return np.sum( np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2) )

class Variance( MinimizationTarget ):
    def __init__( self, data, cnstpow, ds=1 ):
        MinimizationTarget.__init__(self, data, cnstpo, ds=ds )

    def evaluate( self, coeff ):
        Ncoeff = int( len(coeff)/2 )
        coeff = coeff[:Ncoeff] #+ 1j*coeff[Ncoeff:]
        corrected = self.subtractModes( coeff )
        return np.var( corrected[self.sup==1] )
