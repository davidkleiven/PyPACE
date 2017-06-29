import numpy as np
from scipy import ndimage
import cytParallel as cytp
import reconstructor as rec

class Support( object ):
    def __init__( self, mode="real" ):
        self.mode = mode
        if ( mode != "real" and mode != "imag" and mode != "abs" ):
            raise ValueError("Mode has to be either real, imag or abs")

    def get( self, data ):
        """
        Returns a mask with 1 inside the support and zero outside
        """
        NotImplementedError("Child classes has to implement the get function")

class Percentile( Support ):
    def __init__( self, percentile, mode="real" ):
        Support.__init__( self, mode=mode )
        if ( percentile < 0 or percentile > 100 ):
            raise ValueError("The percentile has to be between 0 and 100")
        self.percentile = percentile

    def get( self, data ):
        """
        Set all pixels with intensity above the percentile to 1 in the mask
        """
        mask = np.zeros(data.shape, dtype=np.uint8 )
        if ( self.mode == "real" ):
            threshold = np.percentile( data.real, self.percentile )
            mask[data.real>threshold] = 1
        elif ( self.mode == "imag" ):
            threshold = np.percentile( data.imag, self.percentile )
            mask[data.imag>threshold] = 1
        else:
            threshold = np.percentile( np.abs(data), self.percentile )
            mask[np.abs(data)>threshold] = 1
        return mask

class FractionOfMaxSupport( Support ):
    def __init__( self, fractionOfMax, mode="real" ):
        Support.__init__( self, mode=mode )
        self.fractionOfMax = fractionOfMax

    def get( self, data ):
        """
        Pixels with a value larger than a given fraction of the maximum value are included in the support
        """
        mask = np.zeros(data.shape, dtype=np.uint8 )
        if ( self.mode == "real" ):
            threshold = np.max( data.real )*self.fractionOfMax
            mask[data.real>threshold] = 1
        elif ( self.mode == "imag" ):
            threshold = np.max( data.imag )*self.fractionOfMax
            mask[data.imag>threshold] = 1
        else:
            threshold = np.max( np.abs(data) )*self.fractionOfMax
            mask[np.abs(data)>threshold] = 1
        return mask

class LargerThanFractionAfterGaussianBlur( Support ):
    def __init__( self, fraction, reconstruct, mode="real" ):
        Support.__init__( self, mode=mode)
        if ( not isinstance(reconstruct, rec.Reconstructor) ):
            raise TypeError("Argument reconstruct has to be of type Reconstructor")
        self.width = 3.0
        self.decay = 0.01
        self.fraction = fraction
        self.rec = reconstruct
        self.blurEvery=20

    def get( self, data ):
        """
        Apply gaussian blur and then all pixels with a real part larger than 10 percent of the maxvalue
         is included in the support
        """
        mask = np.zeros(data.shape, dtype=np.uint8 )
        if ( self.mode == "real "):
            if ( self.rec.currentIter%self.blurEvery == 0 ):
                data.real = ndimage.filters.gaussian_filter( data.real, 0.25+self.width )
            #threshold = np.max(data.real)*self.fraction
            threshold = cytp.max(data.real)*self.fraction
            mask = cytp.getThresholdMask( data.real, mask, threshold )
        elif ( self.mode == "imag" ):
            if ( self.rec.currentIter%self.blurEvery == 0 ):
                data.imag = ndimage.filters.gaussian_filter( data.imag, 0.25+self.width )
            threshold = np.max(data.imag)*self.fraction
            mask[data.imag>threshold] = 1
        else:
            if ( self.rec.currentIter%self.blurEvery == 0 ):
                data = ndimage.filters.gaussian_filter( np.abs(data), 0.25+self.width )
            threshold = cytp.max( np.abs(data) )*self.fraction
            mask = cytp.getThresholdMask( np.abs(data), mask, threshold )
            #mask[np.abs(data)>threshold] = 1
        self.width -= self.decay*self.width
        return mask
