import numpy as np
import segmentor as seg
import shellCategorize as kmS
from scipy import optimize as opt

class ShellSegmentor( seg.Segmentor ):
    def __init__( self, data, comm=None ):
        seg.Segmentor.__init__( self, data, comm=comm )
        self.shellradii = None

    def kmeans( self, Nclusters, maxIter=1000 ):
        """
        Perform k-means also taking into account the radial distance from the center
        """
        self.data *= self.data.shape[0]/(2*self.data.max() )
        self.data = np.nan_to_num(self.data)
        self.means = np.linspace( self.data.min(), self.data.max(), Nclusters, endpoint=False )
        rmin = 40.0
        self.shellradii = np.linspace( rmin, self.data.shape[0]/2, Nclusters, endpoint=False )
        dr = self.shellradii[1] - self.shellradii[0]
        self.shellradii += dr
        #weights = np.zeros(2)
        #weights[0] = 1.0
        #weights[1] = 0.0
        #kmS.categorize( self, weights )
        assert( self.data.shape == self.clusters.shape )
        bounds = []
        for i in range(Nclusters):
            bounds.append((rmin,255.0))
        result = opt.minimize( self.shellStandardDev, self.shellradii, bounds=bounds )
        self.shellradii = result["x"]


    def shellStandardDev( self, radii ):
        self.shellradii = np.sort(radii)
        return kmS.avgShellStdDev( self )
