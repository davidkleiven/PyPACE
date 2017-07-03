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

class SliceDensityCorrector( dc.DensityCorrector ):
    def __init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize, comm=None, debug=False,
    projectionAxis=2 ):
        dc.DensityCorrector.__init__( self, reconstructedFname, kspaceFname, wavelength, voxelsize, comm=None, debug=False )

        self.computeMask()

        # Compute the q-weighting of the data
        self.qweight.compute( showPlot=False )

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

        self.kspaceSum = np.sum(self.sliceKspace)

    def plotSliceKspace( self, fig=None ):
        if ( fig is None ):
            fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow( self.sliceKspace, cmap="nipy_spectral", norm=mpl.colors.LogNorm() )
        return fig

    def buildKspace( self ):
        ks = np.zeros(self.sliceKspace.shape)
        start = int( ks.shape[0]/4 )
        end = int( 3*ks.shape[0]/4 )
        for i in range( len(self.segmentor.means) ):
            #print (self.segmentor.projectedClusters[i].density)
            ks[start:end,start:end] += self.segmentor.projectedClusters[i].density*self.segmentor.means[i]

        wavenumber = 2.0*np.pi/self.wavelength
        ks = np.exp(1j*ks*wavenumber*self.voxelsize)-1.0
        ff = np.abs( np.fft.fft( ks ) )**2
        ff = np.fft.fftshift(ff)
        ff = self.qweight.weightData( ff )
        ff *= self.kspaceSum/ff.sum()
        return ff

    def residual( self, x ):
        self.segmentor.means[1:] = x
        ff = self.buildKspace()
        return pcmp.maskedSumOfSquares( self.sliceKspace, ff, self.projMask )

    def fit( self, nClusters=6, maxDelta=1E-4 ):
        self.segment( 6 )
        self.segmentor.projectClusters()
        x0 = np.random.rand( len(self.segmentor.means)-1 )*maxDelta
        x = opt.least_squares( self.residual, x0, bounds=(0.0,maxDelta) )
        print (x)


    def saveAllSliceClusters( self ):
        self.segmentor.plotAllSlices()
