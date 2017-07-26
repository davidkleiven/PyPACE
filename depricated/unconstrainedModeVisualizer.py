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
        mlab.pipeline.contour_surface( src, opacity=0.3 )

    def plotMask( self, maxVoxels=32 ):
        fig = mlab.figure( bgcolor=(0,0,0) )
        ratio = 1
        if ( self.cnstpow.mask.shape[0] > maxVoxels ):
            ratio = int( self.cnstpow.mask.shape[0]/maxVoxels )
            print ("Downsampling mask with a factor of %d before visualizing with Mayavi"%(ratio))

        src = mlab.pipeline.scalar_field( 1-self.cnstpow.mask[::ratio,::ratio,::ratio] )
        mlab.pipeline.contour_surface( src, opacity=0.8 )

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

    def plotMode2DReal( self, modeNum, maxVoxels=32 ):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        ratio = int( self.cnstpow.support.shape[0]/maxVoxels )
        sup = self.cnstpow.support[::ratio,::ratio,::ratio]
        mask = self.cnstpow.mask[::ratio,::ratio,::ratio]
        N = self.cnstpow.support.shape[0]
        x = np.linspace( -N*self.cnstpow.voxelsize/2, N*self.cnstpow.voxelsize/2, N/ratio )
        y = np.linspace( -N*self.cnstpow.voxelsize/2, N*self.cnstpow.voxelsize/2, N/ratio )
        z = np.linspace( -N*self.cnstpow.voxelsize/2, N*self.cnstpow.voxelsize/2, N/ratio )
        X,Y= np.meshgrid(x,y)
        mode = np.abs( self.cnstpow.getEigenModeReal( modeNum, X, Y, 0.0 ) )
        N = sup.shape[0]

        center = int( N/2 )
        ax1.imshow( mode, cmap="inferno")
        ax1.imshow( sup[center,:,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax2 = fig.add_subplot(2,3,2)
        mode = np.abs( self.cnstpow.getEigenModeReal( modeNum, X, 0.0, Y ) )
        ax2.imshow( mode, cmap="inferno" )
        ax2.imshow( sup[:,center,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax3 = fig.add_subplot(2,3,3)
        mode = np.abs( self.cnstpow.getEigenModeReal( modeNum, 0.0, X, Y ) )
        ax3.imshow( mode, cmap="inferno" )
        ax3.imshow( sup[:,:,center], cmap="bone", alpha=0.3, interpolation="none" )

        del X,Y
        #mask = self.cnstpow.mask
        N = self.cnstpow.mask.shape[0]
        kx = np.linspace( -np.pi/self.cnstpow.voxelsize, np.pi/self.cnstpow.voxelsize, N/ratio )
        N = mask.shape[0]
        KX,KY = np.meshgrid(kx,kx)

        #mode = np.abs( self.cnstpow.getEigenModeFourier( modeNum, KX, KY, KZ) )
        center = int( N/2 )
        ax4 = fig.add_subplot(2,3,4)
        mode = np.abs( self.cnstpow.getEigenModeFourier( modeNum, KX, KY, 0.0 ) )
        ax4.imshow( mode, cmap="inferno" )
        ax4.imshow( 1-mask[center,:,:], cmap="bone", alpha=0.3, interpolation="none")
        ax5 = fig.add_subplot(2,3,5)
        mode = np.abs( self.cnstpow.getEigenModeFourier( modeNum, KX, 0.0, KY ) )
        ax5.imshow( mode, cmap="inferno")
        ax5.imshow( 1-mask[:,center,:], cmap="bone", alpha=0.3, interpolation="none" )
        ax6 = fig.add_subplot(2,3,6)
        mode = np.abs( self.cnstpow.getEigenModeFourier( modeNum, 0.0, KX, KY ) )
        ax6.imshow( mode, cmap="inferno")
        ax6.imshow( 1-mask[:,:,center], cmap="bone", alpha=0.3, interpolation="none" )
        return fig
