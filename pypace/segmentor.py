import sys
sys.path.append( "./" )
import numpy as np
import categorize as catg

class Segmentor(object):
    def __init__( self, data, comm=None ):
        self.data = data
        self.means = None
        self.clusters = np.zeros( self.data.shape, dtype=np.uint8 )
        self.comm = comm

    def _clusterID( self, newvalue ):
        return np.argmin( np.abs(newvalue-means) )

    def kmeans( self, Nclusters, maxIter=1000 ):
        """
        Apply k-means clustering to the data
        """
        self.means = np.linspace( self.data.min(), self.data.max(), Nclusters, endpoint=False )
        for i in range(maxIter):
            converged = catg.categorize( self )
            if ( converged ):
                if ( self.comm is None ):
                    print ("K-means converged in %d iterations"%(i+1))
                else:
                    if ( self.comm.Get_rank() == 0 ):
                        print ("K-means converged in %d iterations"%(i+1))
                return
        print ("Warning! Max number of iterations in the kmeans was reached")

    def replaceDataWithMeans( self ):
        """
        Replacte the data values with the mean of the cluster it belongs to
        """
        for i in range(len(self.means) ):
            self.data[self.clusters==i] = self.means[i]

    def getSingleCluster( self, clusterIndx ):
        data = np.zeros(self.data.shape)
        data[self.clusters==clusterIndx] = 1
        return data
