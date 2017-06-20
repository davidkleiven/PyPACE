import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

class DensityCorrector(object):
    def __init__( self, reconstructedFname, kspaceFname ):
        self.reconstructed = np.load( reconstructedFname )
        self.kspace = np.load( kspaceFname )

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

    def plotKspace( self, cmap="inferno" ):
        """
        Plots cuts in the kspace scattering pattern of the object
        Returns:
        fig, [ax1,ax2,ax3]
        fig: Figure object
        [ax1,ax2,ax3]: array of the three ax objects describing the subplots
        """
        assert( len(self.kspace.shape) == 3 )
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        centerX = int(self.kspace.shape[0]/2)
        centerY = int(self.kspace.shape[1]/2)
        centerZ = int(self.kspace.shape[2]/2)

        ax1.imshow( self.kspace[:,:,centerZ], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax1.set_title("$k_xk_y$-plane")

        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( self.kspace[:,centerY,:], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax2.set_title("$k_xk_z$-plane")

        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow( self.kspace[centerX,:,:], cmap=cmap, norm=mpl.colors.LogNorm() )
        ax3.set_title("$k_yk_z$-plane")
        return fig, [ax1,ax2,ax3]
