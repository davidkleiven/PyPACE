import sys
sys.path.append(".")
import numpy as np
import supports as sup
import cytParallel as cytp

class FourierConstraint:
    def __init__( self, measuredScat ):
        self.measured = np.sqrt( np.abs(measuredScat) )
        self.measured = np.fft.ifftshift(self.measured)

    def apply( self, data ):
        #mask = np.zeros(self.measured.shape, dtype=np.uint8 )
        #mask[self.measured>1E-18] = 1
        #data[mask==1] = data[mask==1]*self.measured[mask==1]/np.abs(data[mask==1])
        #data = np.nan_to_num( data )
        cytp.applyFourierConstraint( data, self.measured )
        return data

class RealSpaceConstraint(object):
    def __init__(self, support ):
        self.support = support

    def apply( self, data ):
        raise NotImplementedError("Child classes need to implement the apply function")

class SignFlip( RealSpaceConstraint ):
    def __init__( self, support ):
        RealSpaceConstraint.__init__( self, support )

    def apply( self, data ):
        #img = np.abs(data)
        #perc = np.percentile(img,75)
        #self.support.mask[img>perc] = 1
        #self.support.mask[img<=perc] = 0
        data[self.support.mask==0] = -data[self.support.mask==0]
        return data

class Hybrid( RealSpaceConstraint ):
    def __init__( self, beta, lastObject, support ):
        RealSpaceConstraint.__init__( self, support )
        self.beta = beta
        self.lastObject = lastObject

    def apply( self, data ):
        #mask = self.support.get(data)
        #mask[data.real<0.0] = 0
        #data[mask==0] = self.lastObject[mask==0] - self.beta*data[mask==0]
        #data[mask==0] -= self.beta*self.lastObject[mask==0]
        cytp.applyHybridConstraint( data, self.support.mask, self.beta, self.lastObject )
        return data

class FixedSupport( RealSpaceConstraint ):
    def __init__( self, support ):
        RealSpaceConstraint.__init__( self, support )

    def apply( self, data ):
        data[self.support.mask==0] = 0.0
        return data
