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
    """
    Class that segments an object into clusters based on the voxel value

    data: ndarray
        Array to be segmented

    comm: MPI.Comm
        MPI communicator object. If not given the process will run a single processor
    """
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

        Nclusters: int
            Number of clusters

        maxIter: int
            Maximum number of iterations
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

        width: int
            Width of the cubic region in voxels
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
        """
        clusterIndx: int
            ID of the cluster to return
        Returns: ndarray
            3D array of voxels that belong to the same cluster
        """
        data = np.zeros(self.data.shape)
        data[self.clusters==clusterIndx] = 1
        return data

    def projectClusters( self, axis=2 ):
        """
        Project clusters along an axis

        axis: int
            Axis to project along. Has to be 0,1 or 2
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
        """
        Plot cut planes through all clusters
        """
        if ( len(self.projectedClusters) == 0 ):
            print ("There are no clusters. Have the 3D matrix been reconstructed yet?")
        for i in range(len(self.projectedClusters)):
            fig = self.projectedClusters[i].plot()
            fig.savefig("figures/azm%d.png"%(i))

    def plotCluster( self, clusterID, downsample=4 ):
        """
        Plot single cluster using Mayavi

        clusterID: int
            ID of the cluster to plot

        downsample: int
            Downsampling factor. An a standard computer Mayavi runs slow for large datasets.
            Hence, it may be an advatage to reduce the array before plotting it
        """
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
    """
    Class containing a projected cluster

    id: int
        ID of the cluster
    """
    def __init__( self, id ):
        self.density = None
        self.id = id

    def project( self, clusters, axis=2 ):
        """
        Project cluster along axis

        clusters: ndarray
            Array containning the cluster ID of each voxel

        axis: int
            Axis to project along. Has to be 0,1 or 2
        """
        assert( clusters.shape[0] == clusters.shape[1] )
        assert( clusters.shape[0] == clusters.shape[2] )
        N = clusters.shape[0]
        self.density = np.zeros( (N,N), dtype=np.uint8 )
        self.density = pcmp.projectCluster( clusters, self.id, self.density, axis )

    def plot( self, fig=None ):
        """
        Plot the azimuthally averaged domain

        fig: Matplotlib figure
            If given the plot will be added to the figure, otherwise a new figure is created

        Returns: fig
            Instance of the figure
        """
        if ( fig is None ):
            fig = plt.figure()
        ax = fig.add_subplot( 1,1,1 )
        ax.imshow( self.density, cmap="bone", interpolation="none" )
        return fig
