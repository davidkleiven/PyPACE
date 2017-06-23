from __future__ import print_function
import config
import numpy as np
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import segmentor as seg
import qWeighting as qw
import projectionApprox as pa
from scipy.ndimage import interpolation as sciinterp
import multiprocessing as mp
import geneticAlgorithm as ga
from mpi4py import MPI
import pickle as pck


class DensityCorrector(object):
    def __init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize, comm=None, debug=False ):
        self.reconstructed = np.load( reconstructedFname ).astype(np.float64)
        self.kspace = np.load( kspaceFname )
        self.kspaceIntegral = self.kspace.sum()
        self.segmentor = seg.Segmentor(self.reconstructed, comm)
        self.qweight = qw.Qweight( self.kspace )
        self.projector = pa.ProjectionPropagator( self.reconstructed, wavelength, voxelsize, kspaceDim=self.kspace.shape[0] )
        self.newKspace = None
        self.optimalRotationAngleDeg = 0
        self.hasOptimizedRotation = False
        self.comm = comm
        self.ga = None
        self.debug = debug
        self.mask = np.zeros(self.kspace.shape, dtype=np.uint8 )


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

    def computeMask( self ):
        """
        Computes a mask where all the points that are not measured in the
        experimental dataset is set to 0 and all points that are included
        are set to 1
        """
        self.mask = np.zeros(self.kspace.shape, dtype=np.uint8 )
        self.mask[self.kspace > 10.0*self.kspace.min()] = 1

    def getMeanSqError( self, angle ):
        """
        Returnes the mean square error between the simulated and the measured dataset
        This function is only used to se if the simulated pattern is rotated with respect to the measured.
        It is not used in the fitting procedure and thus a very accurate mean square error is not needed
        Hence, all the arrays are downsampled by a factor 4 to speed up the determination of the
        overall rotation angle
        """
        ds = 4
        rotated = sciinterp.rotate( self.newKspace[::ds,::ds,::ds], angle, axes=(1,0), reshape=False)
        return np.sum( (rotated-self.kspace[::ds,::ds,::ds])**2 )

    def optimizeRotation( self, nangles=24 ):
        """
        Computes the rotation angle between the measured and the simulated scattering pattern
        """
        angle = np.linspace(0,180,nangles)
        meanSquareError = np.zeros(len(angle))
        if ( self.comm is None ):
            anglesPerProc = len(angle)
            rank = 0
        else:
            anglesPerProc = int( len(angle)/self.comm.size )
            rank = self.comm.Get_rank()
        upper = anglesPerProc*rank+anglesPerProc
        if ( upper >= len(angle) ):
            upper = len(angle)
        for i in range(anglesPerProc*rank, upper):
            meanSquareError[i] = self.getMeanSqError(angle[i])

        if ( not self.comm is None ):
            dest = np.zeros(len(meanSquareError))
            self.comm.Reduce( meanSquareError, dest, op=MPI.SUM, root=0 )
            if ( self.comm.Get_rank() == 0 ):
                meanSquareError = dest
        meanSquareError = np.sqrt(meanSquareError)
        meanSquareError /= (self.kspace.shape[0]**3)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(angle,meanSquareError, color="black")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Mean square error")
        self.optimalRotationAngleDeg = angle[np.argmin(meanSquareError)]
        return fig, ax


    def buildKspace( self, angleStepDeg ):
        """
        Compute 3D scattering pattern using the projection approximation
        """
        if ( not self.qweight.weightsAreComputed ):
            self.qweight.compute()
            self.qweight.weightData( self.kspace )

        self.newKspace = self.projector.generateKspace( angleStepDeg )
        self.qweight.weightData( self.newKspace )
        self.newKspace *= self.kspaceIntegral/self.newKspace.sum()
        if ( not self.hasOptimizedRotation ):
            self.optimizeRotation()
            self.hasOptimizedRotation = True
            self.computeMask()
        self.newKspace = sciinterp.rotate( self.newKspace, self.optimalRotationAngleDeg, axes=(1,0), reshape=False)
        self.newKspace[self.mask==0] = np.nan

    def costFunction( self ):
        """
        Return the mean square error between the simulated and the measured scattering mattern
        All points that are not included in the experimental dataset are masked out
        """
        normalization = np.sum( self.newKspace[self.mask==1] )
        return np.sqrt( np.sum( (self.newKspace[self.mask==1]-self.kspace[self.mask==1])**2 ) )/normalization

    def fit( self, nClusters, angleStepKspace=10.0, maxDelta=1E-4, nGAgenerations=50, printStatusMessage=True ):
        """
        Fit the simulated scattering pattern to the experimental by using the Genetic Algorithm
        """
        self.segment( nClusters )
        self.ga = ga.GeneticAlgorithm( self, maxDelta, self.comm, nGAgenerations, debug=self.debug )
        self.ga.printStatusMessage = printStatusMessage
        self.ga.run( angleStepKspace )
