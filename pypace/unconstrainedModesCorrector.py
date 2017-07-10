import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndimg
from scipy import optimize as opt

class UnconstrainedModeCorrector( object ):
    def __init__( self, cnstpow, data, minimizer="laplacian" ):
        self.cnstpow = cnstpow
        ratio = int(data.shape[0]/self.cnstpow.support.shape[0])
        if ( ratio != 1 ):
            print ("Downsampling the data by a factor %d"%(ratio))
            self.data = data[::ratio,::ratio,::ratio]

        if ( minimizer == "laplacian"):
            self.minimizer = LaplacianSquared( self.data, self.cnstpow )
        elif ( minimizer == "gradient" ):
            self.minimizer = GradientSquared( self.data, self.cnstpow )
        else:
            raise ValueError("minimizer has to be laplacian or gradient")

    def correct( self, threshold ):
        if ( threshold > 1.0 or threshold < 0.0 ):
            raise ValueError("Threshold has to be in the range [0,1]")

        # Count the number of modes to include
        nModes = -1
        for i in range(len(self.cnstpow.eigval) ):
            if ( self.cnstpow.eigval[i] > threshold ):
                nModes = i
                break

        if ( nModes < 0 ):
            nModes = len(self.cnstpow.eigval)
        self.minimizer.nModes = nModes
        print ( "Using %d least constrained modes"%(nModes) )

        self.minimizer.preCalculateEigenmodes()

        print ("Optimizing results")

        res = opt.minimize( self.minimizer.evaluate, np.ones(nModes) )
        x = np.array( res["x"] )
        self.data = self.minimizer.subtractModes( x )
        print (res["message"])


class MinimizationTarget( object ):
    def __init__( self, data, cnstpow ):
        self.data = data
        self.cnstpow = cnstpow
        self.nModes = 0
        self.eigmodes = []

    def preCalculateEigenmodes( self ):
        N = self.data.shape[0]
        x = np.linspace(-N/2,N/2,N)
        X,Y,Z = np.meshgrid( x, x, x )
        for i in range(self.nModes):
            neweigmode = self.cnstpow.getEigenModeReal( i, X,Y,Z )
            self.eigmodes.append(neweigmode)

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
    def __init__( self, data, cnstpow ):
        MinimizationTarget.__init__( self, data, cnstpow )

    def evaluate( self, coeff ):
        corrected = self.subtractModes( coeff )
        laplace = ndimg.filters.laplace( np.abs(corrected), mode="constant", cval=0.0 )
        return np.sum( laplace**2 )

class GradientSquared( MinimizationTarget ):
    def __init__( self, data, cnstpow ):
        MinimizationTarget.__init__( self, data, cnstpow )

    def evaluate( self, coeff ):
        corrected = np.abs( self.subtractModes( coeff ) )
        grad = np.gradient( corrected )
        return np.sum( np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2) )
