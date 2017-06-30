import numpy as np
from scipy import ndimage
import cytParallel as cytp
import reconstructor as rec

class Support( object ):
    def __init__( self, mask, fraction ):
        self.mask = mask
        self.width = 5
        self.decay = 0.01
        self.fraction = fraction

    def update( self, data ):
        """
        Updates the support
        """
        img = np.empty(data.shape)
        img = cytp.modulus(data, img)
        img = ndimage.filters.gaussian_filter( img, 0.25+self.width )
        self.width -= self.decay*self.width
        threshold = cytp.max(img)*self.fraction
        self.mask[:,:,:] = cytp.getThresholdMask( img, self.mask, threshold )
