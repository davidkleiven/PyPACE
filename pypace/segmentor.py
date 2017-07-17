import sys
sys.path.append( "./" )
import numpy as np
import categorize as catg
import pypaceCython as pcmp
import multiprocessing as mp
from matplotlib import pyplot as plt

try:
    from mayavi import mlab
    haveMayavi = True
except ImportError as exc:
    print (str(exc))
    haveMayavi = False

class Segmentor(object):
    def __init__( self, data, comm=None ):
        self.data = data
        self.means = None
        self.clusters = np.zeros( self.data.shape, dtype=np.uint8 )
        self.comm = comm
        self.projectedClusters = []

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

    def createSeparateClusterCenter( self, width ):
        """
        Extracts the center of the object and creates a separate cluster for it
        """
        maxID = len(self.means)
        center = int( self.clusters.shape[0]/2 )
        delta = int(width/2)
        self.clusters[center-delta:center+delta,center-delta:center+delta,center-delta:center+delta] = maxID
        self.means = np.append(self.means, self.means[-1] )


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

    def projectClusters( self, axis=2 ):
        """
        Perform azimuthal average of all the clusters
        """
        self.projectedClusters = []
        mpArgs = []
        for clusterID in range(len(self.means)):
            proj = ProjectedCluster(clusterID)
            proj.project( self.clusters, axis )
            self.projectedClusters.append(proj)
            #mpArgs.append([proj,axis])

        #workers = mp.Pool( processes=mp.cpu_count() )
        #workers.map( self._projParallel, mpArgs )

    def _projParallel( self, args ):
        args[0].project( self.clusters, axis=args[1] )

    def plotAllSlices( self ):
        if ( len(self.projectedClusters) == 0 ):
            print ("There are no clusters. Have the 3D matrix been reconstructed yet?")
        for i in range(len(self.projectedClusters)):
            fig = self.projectedClusters[i].plot()
            fig.savefig("figures/azm%d.png"%(i))

    def plotCluster( self, clusterID, downsample=4 ):
        if ( not haveMayavi ):
            return
        fig = mlab.figure( bgcolor=(0,0,0) )
        mask = np.zeros( self.clusters.shape, dtype=np.uint8 )
        mask[self.clusters==clusterID] = 1
        # Downsample by summing
        Nx = mask.shape[0]
        Ny = mask.shape[1]
        Nz = mask.shape[2]
        dsMask = np.zeros( (int(Nx/downsample),Ny,Nz), dtype=np.uint8)
        current = 0
        for i in range(0,dsMask.shape[0]):
            dsMask[i,:,:] = np.sum(mask[current:current+downsample,:,:],axis=0)
            current += downsample

        mask = dsMask
        current = 0
        dsMask = np.zeros( (int(Nx/downsample), int(Ny/downsample), Nz), dtype=np.uint8 )
        for i in range(0,dsMask.shape[1]):
            dsMask[:,i,:] = np.sum( mask[:,current:current+downsample,:], axis=1 )
            current += downsample
        mask = dsMask
        current = 0
        dsMask = np.zeros( (int(Nx/downsample), int(Ny/downsample), int(Nz/downsample)), dtype=np.uint8 )
        for i in range(0,dsMask.shape[2]):
            dsMask[:,:,i] = np.sum(mask[:,:,current:current+downsample],axis=2)
            current += downsample

        mask = dsMask
        #mask = mask[::downsample,::downsample,::downsample]
        #if ( mask.max() != 1 ):
        #    return
        src = mlab.pipeline.scalar_field(mask)
        #vol = mlab.pipeline.volume( src )
        #mlab.pipeline.threshold( vol, low=0.5 )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="x_axes")
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="y_axes")
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="z_axes")
        mlab.show()

class ProjectedCluster( object ):
    def __init__( self, id ):
        self.density = None
        self.id = id

    def project( self, clusters, axis=2 ):
        """
        Performs azimuthal average when rotating around the axis specified by the axis argument
        """
        assert( clusters.shape[0] == clusters.shape[1] )
        assert( clusters.shape[0] == clusters.shape[2] )
        N = clusters.shape[0]
        self.density = np.zeros( (N,N), dtype=np.uint8 )
        self.density = pcmp.projectCluster( clusters, self.id, self.density, axis )

    def plot( self, fig=None ):
        """
        Plot the azimuthally averaged domain
        """
        if ( fig is None ):
            fig = plt.figure()
        ax = fig.add_subplot( 1,1,1 )
        ax.imshow( self.density, cmap="bone", interpolation="none" )
        return fig
