import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import segmentor as seg
import qWeighting as qw
import projectionApprox as pa


class DensityCorrector(object):
    def __init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize ):
        self.reconstructed = np.load( reconstructedFname ).astype(np.float64)
        self.kspace = np.load( kspaceFname )
        self.segmentor = seg.Segmentor(self.reconstructed)
        self.qweight = qw.Qweight( self.kspace )
        self.projector = pa.ProjectionPropagator( self.reconstructed, wavelength, voxelsize )
        self.newKspace = None

    def plotRec( self, show=False, cmap="inferno" ):
        """
        Plots cuts in the reconstructed object through the center
        Returns:
        fig, [ax1,ax2,ax3]
        fig: Figure object
        [ax1,ax2,ax3]: array of the three ax objects describing the subplots
        """
        assert( len(self.reconstructed.shape) == 3 )
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        centerX = int(self.reconstructed.shape[0]/2)
        centerY = int(self.reconstructed.shape[1]/2)
        centerZ = int(self.reconstructed.shape[2]/2)

        ax1.imshow( self.reconstructed[:,:,centerZ], cmap=cmap )
        ax1.set_title("xy-plane")

        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( self.reconstructed[:,centerY,:], cmap=cmap )
        ax2.set_title("xz-plane")

        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow( self.reconstructed[centerX,:,:], cmap=cmap )
        ax3.set_title("yz-plane")
        return fig, [ax1,ax2,ax3]

    def plotKspace( self, data, cmap="inferno" ):
        """
        Plots cuts in the kspace scattering pattern of the object
        Returns:
        fig, [ax1,ax2,ax3]
        fig: Figure object
        [ax1,ax2,ax3]: array of the three ax objects describing the subplots
        """
        assert( len(data.shape) == 3 )
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        centerX = int(data.shape[0]/2)
        centerY = int(data.shape[1]/2)
        centerZ = int(data.shape[2]/2)

        ax1.imshow( data[:,:,centerZ], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax1.set_title("$k_xk_y$-plane")

        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( data[:,centerY,:], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax2.set_title("$k_xk_z$-plane")

        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow( data[centerX,:,:], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax3.set_title("$k_yk_z$-plane")
        return fig, [ax1,ax2,ax3]

    def segment( self, Nclusters, maxIter=1000 ):
        """
        Segments the image into Nclusters
        """
        self.segmentor.kmeans( Nclusters, maxIter=maxIter )

    def plotClusters( self, cluster, cmap="bone" ):
        """
        Plots individual clusters given by the cluster array
        Example:
        this.plotClusters(0)
        """
        fig = plt.figure()
        ax = []
        data = self.segmentor.getSingleCluster( cluster )
        centerX = int(data.shape[0]/2)
        centerY = int(data.shape[1]/2)
        centerZ = int(data.shape[2]/2)
        ax.append( fig.add_subplot(1,3,len(ax)+1))
        ax[-1].imshow(data[:,:,centerZ], cmap=cmap)
        ax.append( fig.add_subplot(1,3,len(ax)+1))
        ax[-1].imshow(data[:,centerY,:], cmap=cmap)
        ax.append( fig.add_subplot(1,3,len(ax)+1))
        ax[-1].imshow(data[centerX,:,:], cmap=cmap)
        fig.suptitle("Cluser %d"%(cluster))
        return fig, ax

    def buildKspace( self, angleStepDeg ):
        self.newKspace = self.projector.generateKspace( angleStepDeg )
