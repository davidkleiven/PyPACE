import numpy as np

class FourierConstraint:
    def __init__( self, measuredScat ):
        self.measured = np.sqrt( np.abs(measuredScat) )

    def apply( self, data ):
        return data*self.measured/np.abs(data)

class RealSpaceConstraint(object):
    def __init__(self):
        pass

    def apply( self, data ):
        raise NotImplementedError("Child classes need to implement the apply function")

class SignFlip( RealSpaceConstraint ):
    def __init__( self, threshold ):
        self.threshold = threshold

    def apply( self, data ):
        mean = np.mean(np.abs(data))
        self.threshold = np.percentile(data.real,75)
        #print (mean,self.threshold)
        data[data.real<self.threshold] = -data[data.real<self.threshold]
        return data

class Hybrid( RealSpaceConstraint ):
    def __init__( self, threshold, beta, lastObject ):
        self.threshold = threshold
        self.beta = beta
        self.lastObject = lastObject

    def apply( self, data ):
        mean = np.mean(data)
        self.threshold = np.percentile(data.real,75)
        data[data.real<self.threshold] = self.lastObject[data.real<self.threshold] - self.beta*data[data.real<self.threshold]
        return data

class FixedSupport( RealSpaceConstraint ):
    def __init__( self, threshold ):
        self.threshold = threshold

    def apply( self, data ):
        mean = np.mean(data)
        self.threshold = np.percentile(data.real,75)
        data[data.real<self.threshold] = 0.0
        return data
