import categorize as ctg
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import categorize as catg

class Qweight(object):
    def __init__( self, kspaceData ):
        self.data = kspaceData
        self.interscept = 1.0
        self.slope = 0.0

    def compute( self, showPlot=False ):
        # Perform a radial average
        qxMax = self.data.shape[0]/2
        qyMax = self.data.shape[1]/2
        qzMax = self.data.shape[2]/2
        rmax = np.sqrt( qxMax**2 + qyMax**2 + qzMax**2 )
        Nbins = int( self.data.shape[0]/4 )
        rbins = np.linspace(0.0, rmax, Nbins)
        radialMean = catg.radialMean( self.data, Nbins )

        # Filter out very small values
        rbins = rbins[radialMean > 1E-6*radialMean.max()]
        radialMean = radialMean[radialMean > 1E-6*radialMean.max()]

        self.slope, self.interscept, rvalue, pvalue, stderr = stats.linregress( np.log(rbins), np.log(radialMean) )
        print (self.slope, self.interscept)

        if ( showPlot ):
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.plot( rbins, radialMean, 'o', color="black")
            ax.plot( rbins, self.getWeight(rbins), "--", color="black")
            ax.set_yscale("log")

            ax2 = fig.add_subplot(1,2,2)
            ax2.plot( rbins, radialMean/self.getWeight(rbins), color="black")

    def getWeight( self, q ):
        """
        Return the weighting factor
        """
        return np.exp(self.interscept)*q**self.slope