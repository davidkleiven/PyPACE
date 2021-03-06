import densityCorrector as dc
import pypaceCython as pcmp
import numpy as np
import config
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
import multiprocessing as mp
from scipy import optimize as opt
from scipy import ndimage as ndimg
import subprocess as sub
import h5py as h5
import time
import datetime as dt

try:
    import pyswarm as pso
    hasPSO = True
except ImportError as exc:
    hasPSO = False

class SliceDensityCorrector( dc.DensityCorrector ):
    def __init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize, comm=None, debug=False,
    projectionAxis=2, segmentation="voxels" ):
        """
        SliceDensityCorrector. Correcting the density obtained from the phase retrieval algorithm
        by using only one slice in Fourier space.

        See the documentation for :class:'DensityCorrector' for explination of the first arguments

        projectionAxis: int
            Axis along which the projection of the object should be computed. Has to 0,1 or 2. Default is 2
        """
        dc.DensityCorrector.__init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize, comm=comm, debug=False,
        segmentation=segmentation )

        self.computeMask()

        # Compute the q-weighting of the data
        self.qweight.compute( showPlot=True )

        # Divide the measured data by the weighting factor
        self.qweight.weightData( self.kspace )
        N = self.kspace.shape[0]
        self.projAxis = projectionAxis
        if ( self.projAxis == 0 ):
            self.sliceKspace = self.kspace[int(self.kspace.shape[0]/2),:,:]
            self.projMask = self.mask[int(self.kspace.shape[0]/2),:,:]
        elif ( self.projAxis == 1 ):
            self.sliceKspace = self.kspace[:,int(self.kspace.shape[1]/2),:]
            self.projMask = self.mask[:,int(self.kspace.shape[1]/2),:]
        elif ( self.projAxis == 2 ):
            self.sliceKspace = self.kspace[:,:,int(self.kspace.shape[2]/2)]
            self.projMask = self.mask[:,:,int(self.kspace.shape[2]/2)]
        else:
            raise ValueError("Projection axis has to be either 0,1 or 2")

        self.blurSigma = 3
        #self.sliceKspace = ndimg.gaussian_filter( self.sliceKspace, self.blurSigma )
        self.kspaceSum = np.sum(self.sliceKspace)
        self.bestFF = None
        self.bestResidual = 1E30
        self.optimizationMode = "local"

    def plotSliceKspace( self, fig=None ):
        """
        Plots the slice of the measured data

        fig: Matplit figure
            If given the plot will be added to the existing figure. Otherwise a new figure is created

        Returns: Matplotlib figure
            Instance of the figure
        """
        if ( fig is None ):
            fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow( self.sliceKspace, cmap="nipy_spectral" )
        return fig

    def needPadding( self ):
        """
        Returns: bool
            True if the real space object is half the size in each dimension of the
            Fourier domain. Hence, it needs to be padded.
            False otherwise.
        """
        shp1 = self.sliceKspace.shape
        shp2 = self.segmentor.projectedClusters[0].density.shape
        pad = False

        if ( shp2[0] != shp2[1] ):
            raise ValueError("The projected clusters need to be square")

        if ( shp2[0] == int(shp1[0]/2) ):
            return True
        elif ( shp2[0] == shp1[0] ):
            return False
        else:
            raise ValueError("The realspace object has to either be half the size of the k-space object or of similar shape as the k-space object")

    def buildKspace( self ):
        """
        Computes the scattering pattern using the projection approximation
        """
        ks = np.zeros(self.sliceKspace.shape)

        if ( self.needPadding() ):
            start = int( ks.shape[0]/4 )
            end = int( 3*ks.shape[0]/4 )
        else:
            start = 0
            end = ks.shape[0]
        for i in range( len(self.segmentor.means) ):
            #print (self.segmentor.projectedClusters[i].density)
            ks[start:end,start:end] += self.segmentor.projectedClusters[i].density*self.segmentor.means[i]

        wavenumber = 2.0*np.pi/self.wavelength
        ks = np.exp(1j*ks*wavenumber*self.voxelsize) - 1.0
        ff = np.abs( np.fft.fft2( ks ) )**2
        ff = np.fft.fftshift(ff)
        ff = self.qweight.weightData( ff )
        ff *= self.kspaceSum/ff.sum()
        return ff

    def residual( self, x ):
        """
        Computes the sum of difference squared between the measured and the simulated data

        x: 1D-array
            Array with the means of the clusters
        """
        self.segmentor.means[1:] = x
        self.segmentor.means[0] = 0.0
        ff = self.buildKspace()
        return pcmp.maskedSumOfSquares( self.sliceKspace, ff, self.projMask )

    def fitSingle( self, maxDelta=1E-4 ):
        """
        Perform local optimization from a random initial condisiont

        maxDelta: float
            Maximum value of the deviation from unity of the real part of the refractive index
        """
        x0 = np.random.rand( len(self.segmentor.means)-1 )*maxDelta
        #x0 = self.segmentor.means*maxDelta/np.max(self.segmentor.means)
        #x0 = x0[1:]
        optimum = opt.least_squares( self.residual, x0, bounds=(0.0,maxDelta) )

        if ( optimum["cost"] < self.bestResidual ):
            self.bestFF = self.buildKspace()
        return optimum

    def fitPSO( self, nClusters, nIter=1000, maxDelta=1E-4 ):
        """
        Perform the curve fit using Particle Swarm Optimization from the pyswarm module

        nClusers: int
            Number of clusters

        nIter: int
            Maximum number of iterations. Default is 1000

        maxDelta: float
            Maximum value of the deviation from unity of the real part of the refractive index
        """
        if ( hasPSO ):
            lb = np.zeros( len(self.segmentor.means)-1 )
            ub = np.zeros( len(self.segmentor.means)-1 )+maxDelta
            xopt, fopt = pso.pso( residualNotInstanceMethod, lb, ub, args=(self,), maxiter=nIter, debug=True )
        else:
            raise ImportError("The module pyswarm was not found")
        self.segmentor.means[1:] = xopt

        # Create a dictionary similar to the ones return by scipys optimization functions
        result = {
        "cost":fopt,
        "success":True
        }
        return result

    def fit( self, nIter=1000, nClusters=6, maxDelta=1E-4, useSeparateClusterAtCenter=False, centerClusterWidth=0, mode="local" ):
        """
        Fit the e-density parameters to the scattering pattern

        nIter: int
            Maximum number of iterations

        nClusters: int
            Number of clusters

        maxDelta: float
            Maximum value of the deviation from unity of the real part of the refractive index

        useSeparateClusterAtCenter: bool
            If True a cubic cluster at the center of the scatterer is introduced

        centerClusterWidth: int
            Width of the cube introduced at then center in voxels.
            If useSeparateClusterAtCenter is False this argument is ignored

        mode: str
            Optimization mode. Has to be either. local or pso.\n
            local - run multiple local optimizations and pick the best result\n
            pso - Use Particle Swarm Optimization scheme
        """
        if ( mode != "local" and mode != "pso" ):
            raise ValueError("mode has to be either local or pso")
        self.optimizationMode = mode
        self.segment( nClusters )
        if ( useSeparateClusterAtCenter ):
            self.segmentor.createSeparateClusterCenter( centerClusterWidth )
        self.removeInternalPointsFromSurroundingCluster()
        self.segmentor.projectClusters()

        if ( self.comm is None ):
            raise Exception("Fitting requires MPI support")

        rank = self.comm.Get_rank()
        nPerProcess = int( nIter/self.comm.size )
        if ( nPerProcess == 0 or mode =="pso" ):
            nPerProcess = 1

        folder = "tmpOptFiles"
        if ( rank == 0 ):
            sub.call(["mkdir","-p","tmpOptFiles"])
        self.comm.Barrier()
        fname = folder+"/optimizationResults%d.h5"%(rank)
        h5file = h5.File( fname, 'w' )
        for i in range(nPerProcess):
            if ( mode == "local" ):
                optimum = self.fitSingle()
            elif ( mode == "pso" ):
                optimum = self.fitPSO( nClusters, nIter=nIter, maxDelta=maxDelta )
            dset = h5file.create_dataset( "rank%d_%d"%(rank,i), data=self.segmentor.means )
            dset.attrs["converged"] = optimum["success"]
            dset.attrs["cost"] = optimum["cost"]
        h5file.close()

    def merge( self, fname="" ):
        """
        Merges the hdf5 files created by each MPI process.
        If the optimization mode is different from local, the function does nothing.

        fname: str
            Filename of the resulting hdf5 file
        """
        if ( self.optimizationMode != "local" ):
            return
        self.comm.Barrier()
        rank = self.comm.Get_rank()
        if ( rank == 0 ):
            if ( fname != "" ):
                h5file = h5.File(fname, 'a')
            else:
                fname = "fittedElectronDensity.h5"
                h5file = h5.File( fname,'w')

            ts = time.time()
            stamp = dt.datetime.fromtimestamp(ts).strftime("%Y-%-m-%d-%H-%M-%S")

            group = h5file.create_group("g%s"%(stamp))
            for i in range(0,self.comm.size):
                fname = "tmpOptFiles/optimizationResults%d.h5"%(i)
                with h5.File(fname,'r') as hf:
                    for dsetname in hf.keys():
                        h5file.copy(hf.get(dsetname), group, name=dsetname)

            # Store all the projected clusters
            dset = h5file.create_dataset( "clusters", data=self.segmentor.clusters )
            dsetK = h5file.create_dataset( "sliceK", data=self.sliceKspace )
            dsetM = h5file.create_dataset( "mask", data=self.projMask )
            dsetFF = h5file.create_dataset( "bestFarField", data=self.bestFF )
            h5file.close()
            print ("Data saved in %s"%(fname))

    def plotFit( self, means, fig=None ):
        """
        Plot the results of the fit

        means: 1D array
            Array of the measn in each cluster (except the cluster of voxels outisde the support)

        fig: Matplotlib figure
            Instance of a matplotlib figure. If given the plot as added to this figure, otherwise a new figure is created

        Returns: fig
            The Matplotlib figure instance
        """
        self.segmentor.means[1:] = means
        ff = self.buildKspace()
        if ( fig is None ):
            fig = plt.figure()

        ax1 = fig.add_subplot(2,3,1)
        ax1.imshow( self.sliceKspace, cmap="nipy_spectral", interpolation="none" )
        ax1.set_title("Measured")

        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow( ff, cmap="nipy_spectral", interpolation="none")
        ax2.set_title("Simulated")

        ax3 = fig.add_subplot(2,3,3)
        diff = np.abs(ff-self.sliceKspace)
        diff[self.projMask==0] = np.nan
        ax3.imshow( diff, cmap="nipy_spectral", interpolation="none")
        ax3.set_title("Difference")

        ax4 = fig.add_subplot(2,3,4)
        center = int( self.sliceKspace.shape[0]/2)
        ax4.plot( self.sliceKspace[center,:], color="#e41a1c")
        ax4.plot( ff[center,:], color="#377eb8")

        ax5 = fig.add_subplot(2,3,5)
        ax5.plot( self.sliceKspace[:,center], color="#e41a1c")
        ax5.plot( ff[:,center], color="#377eb8")

        ax6 = fig.add_subplot(2,3,6)
        ax6.imshow( self.projMask, cmap="bone", interpolation="none")
        ax6.set_title( "Mask" )
        return fig

    def saveAllSliceClusters( self ):
        """
        Save all the clusters
        """
        self.segmentor.plotAllSlices()

# Wrapper of the residual function
def residualNotInstanceMethod( x, instance ):
    return instance.residual(x)
