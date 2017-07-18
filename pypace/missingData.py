import sys
sys.path.append("./")
sys.path.append("pypace")
sys.path.append("../")
import numpy as np
import missingdatac as mdc
import copy
import config
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt

class MissingDataAnalyzer( object ):
    def __init__( self, mask, support ):
        self.mask = mask
        self.support = support
        self.beta = 1.0

    def step( self, current ):
        x = copy.deepcopy( current )
        pa = self.projectA( self.fB(x) )
        pb = self.projectB( self.fA(x) )
        current = current + self.beta*(pa-pb)
        return current

    def fA( self, x ):
        result = x/self.beta
        pa = self.projectA(x) # This modifies x
        return pa - pa/self.beta + result

    def fB( self, x ):
        res = x/self.beta
        pb = self.projectB(x)
        return pb + pb/self.beta - res

    def projectB( self, x ):
        """
        Make x-consistent with the Fourier domain constraint
        """
        ft = np.fft.rfftn( x )
        mdc.applyFourier( self, x )
        return np.fft.irfftn(x, x.shape )

    def projectA( self, x ):
        """
        Make x-consistent with the real space constraint
        """
        mdc.applyRealSpace( self, x )
        return x

    def plot( self, x ):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        center = int( self.support.shape[0]/2 )
        ax1.imshow( x[center,:,:], cmap="inferno")
        ax1.imshow( self.support[center,:,:], cmap="bone", alpha=0.3 )

        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow( x[:,center,:], cmap="inferno")
        ax2.imshow( self.support[:,center,:], cmap="bone", alpha=0.3)

        ax3 = fig.add_subplot(2,3,3)
        ax3.imshow( x[:,:,center], cmap="inferno")
        ax3.imshow( self.support[:,:,center], cmap="bone", alpha=0.3)
        return fig

    def solve( self, niter=1000 ):
        current = self.support
        for i in range(niter):
            print ("Iteration %d of maximum %d"%(i,niter))
            current = self.step( current )
            self.plot( current )
            plt.show()
