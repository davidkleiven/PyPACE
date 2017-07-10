import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
import h5py as h5

class UnconstrainedModeVisualizer( object ):
    def __init__( self, cnstpow ):
        self.cnstpow = cnstpow

    def saveMaskToHDF5( self ):
        with h5.File("data/mask.h5", 'w') as hf:
            hf.create_dataset( "mask", data=255*self.cnstpow.mask )

    def plotSupport( self ):
        fig = mlab.figure( bgcolor=(0,0,0) )
        src = mlab.pipeline.scalar_field( self.cnstpow.support )
        #vol = mlab.pipeline.volume( src )
        #mlab.pipeline.threshold( vol, low=0.5 )
        mlab.pipeline.contour_surface( src, opacity=0.3 )

    def plotMask( self ):
        fig = mlab.figure( bgcolor=(0,0,0) )
        src = mlab.pipeline.scalar_field( 1-self.cnstpow.mask )
        #vol = mlab.pipeline.volume( src )
        #mlab.pipeline.threshold( vol, low=0.95 )
        mlab.pipeline.contour_surface( src, opacity=0.8 )
        #mlab.pipeline.iso_surface( src )

    def plotModeReal( self, modeNum ):
        N = 32
        x = np.linspace( -2.0*self.cnstpow.scaleX, 2.0*self.cnstpow.scaleX, N )
        y = np.linspace( -2.0*self.cnstpow.scaleY, 2.0*self.cnstpow.scaleY, N )
        z = np.linspace( -2.0*self.cnstpow.scaleZ, 2.0*self.cnstpow.scaleZ, N )
        X,Y,Z = np.meshgrid(x,y,z)
        mode = np.abs( self.cnstpow.getEigenModeReal( modeNum, X, Y, Z ) )

        fig = mlab.figure( bgcolor=(0,0,0) )
        src = mlab.pipeline.scalar_field( mode )
        mlab.pipeline.scalar_cut_plane( src, colormap="plasma" )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="y_axes", colormap="plasma" )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="z_axes", colormap="plasma" )

    def plotMode2DReal( self, modeNum ):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        N = self.cnstpow.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)
        mode = np.abs( self.cnstpow.getEigenModeReal( modeNum, X, Y, Z ) )

        center = int( N/2 )
        ax1.imshow( mode[center,:,:], cmap="inferno")
        ax1.imshow( self.cnstpow.support[center,:,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow( mode[:,center,:], cmap="inferno" )
        ax2.imshow( self.cnstpow.support[:,center,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax3 = fig.add_subplot(2,3,3)
        ax3.imshow( mode[:,:,center], cmap="inferno" )
        ax3.imshow( self.cnstpow.support[:,:,center], cmap="bone", alpha=0.3, interpolation="none" )

        del X,Y,Z
        N = self.cnstpow.mask.shape[0]
        kx = np.linspace(-np.pi,np.pi,N)
        KX,KY,KZ = np.meshgrid(kx,kx,kx)

        mode = np.abs( self.cnstpow.getEigenModeFourier( modeNum, KX, KY, KZ) )
        center = int( N/2 )
        ax4 = fig.add_subplot(2,3,4)
        ax4.imshow( mode[center,:,:], cmap="inferno" )
        ax4.imshow( 1-self.cnstpow.mask[center,:,:], cmap="bone", alpha=0.3, interpolation="none")
        ax5 = fig.add_subplot(2,3,5)
        ax5.imshow( mode[:,center,:], cmap="inferno")
        ax5.imshow( 1-self.cnstpow.mask[:,center,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax6 = fig.add_subplot(2,3,6)
        ax6.imshow( mode[:,:,center], cmap="inferno")
        ax6.imshow( 1-self.cnstpow.mask[:,:,center], cmap="bone", alpha=0.3, interpolation="none" )
        return fig
