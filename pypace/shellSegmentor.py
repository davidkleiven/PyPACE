import numpy as np
import segmentor as seg
import shellCategorize as kmS

class ShellSegmentor( seg.Segmentor ):
    def __init__( self, data, comm=None ):
        seg.Segmentor.__init__( self, data, comm=comm )
        self.shellradii = None

    def kmeans( self, Nclusters, maxIter=1000 ):
        """
        Perform k-means also taking into account the radial distance from the center
        """
        self.data *= self.data.shape[0]/(2*self.data.max() )
        self.means = np.linspace( self.data.min(), self.data.max(), Nclusters, endpoint=False )
        self.shellradii = np.linspace( 0, self.data.shape[0]/2, Nclusters, endpoint=False )
        weights = np.zeros(2)
        weights[0] = 1.0
        weights[1] = 0.0
        kmS.categorize( self, weights )
