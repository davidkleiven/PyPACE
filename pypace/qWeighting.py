import categorize as ctg
import config
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import categorize as catg

class Qweight(object):
    def __init__( self, kspaceData ):
        self.data = kspaceData
        self.interscept = 1.0
        self.slope = 0.0
        self.weightsAreComputed = False
        self.gaussianSlope = 0.0
        self.gaussianInterscept = 0.0
        self.gaussianFitted = False

    def weightData( self, data ):
        if ( not self.weightsAreComputed ):
            raise RuntimeError("Power law fit has not been performed")

        if ( len(data.shape) == 3 ):
            return catg.performQWeighting( data, np.exp(self.interscept), self.slope )
        elif ( len(data.shape) == 2 ):
            return catg.performQWeighting2D( data, np.exp(self.interscept), self.slope )
        else:
            raise TypeError("Data has to be numpy array of dimension 2 or 3")

    def compute( self, showPlot=False ):
        # Perform a radial average
        #qxMax = self.data.shape[0]/2
        #qyMax = self.data.shape[1]/2
        #qzMax = self.data.shape[2]/2
        #rmax = np.sqrt( qxMax**2 + qyMax**2 + qzMax**2 )
        #Nbins = int( self.data.shape[0]/4 )
        #rbins = np.linspace(0.0, rmax, Nbins)
        rbins = self.getRadialBins()
        radialMean = catg.radialMean( self.data, len(rbins) )

        # Filter out very small values
        rbins = rbins[radialMean > 1E-6*radialMean.max()]
        radialMean = radialMean[radialMean > 1E-6*radialMean.max()]
        dr = rbins[1]-rbins[0]
        rbins += dr/2.0

        self.slope, self.interscept, rvalue, pvalue, stderr = stats.linregress( np.log(rbins), np.log(radialMean) )
        self.weightsAreComputed = True

        if ( showPlot ):
            print ("Exponent: ", self.slope)
            print ("Prefactor: ", np.exp(self.interscept))
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.plot( rbins, radialMean, 'o', color="black")
            ax.plot( rbins, self.getWeight(rbins), "--", color="black")
            ax.set_yscale("log")

            ax2 = fig.add_subplot(1,2,2)
            ax2.plot( rbins, radialMean/self.getWeight(rbins), color="black")
            plt.show()

    def getWeight( self, q ):
        """
        Return the weighting factor
        """
        return np.exp(self.interscept)*q**self.slope

    def getRadialBins( self ):
        qxMax = self.data.shape[0]/2
        qyMax = self.data.shape[1]/2
        qzMax = self.data.shape[2]/2
        rmax = np.sqrt( qxMax**2 + qyMax**2 + qzMax**2 )
        Nbins = int( self.data.shape[0]/4 )
        rbins = np.linspace(0.0, rmax, Nbins)
        return rbins

    def fitRadialGaussian( self, showPlot=True ):
        """
        Perform a Gaussian fit to radial averaged pattern
        """
        rbins = self.getRadialBins()
        radialMean = catg.radialMean( self.data, len(rbins) )

        rbins = rbins[radialMean > 1E-6*radialMean.max()]
        radialMean = radialMean[radialMean > 1E-6*radialMean.max()]
        dr = rbins[1]-rbins[0]
        rbins += dr/2.0

        self.gaussianSlope, self.gaussianInterscept, rvalue, pvalue, stderr = stats.linregress( rbins**2, np.log(radialMean) )
        self.gaussianFitted = True

    def radialGaussian( self, r ):
        if ( not self.gaussianFitted ):
            self.fitRadialGaussian( showplot=False )
        return np.exp(self.gaussianInterscept)*np.exp(self.gaussianSlope*r**2)

    def fillMissingDataWithGaussian( self, mask ):
        self.fitRadialGaussian()
        N = self.data.shape[0]
        x = np.linspace(-N/2, N/2, N )
        X,Y,Z = np.meshgrid( x,x,x )
        R = np.sqrt(X**2 + Y**2 + Z**2 )
        del X,Y,Z
        gauss = self.radialGaussian(R)
        del R
        self.data[mask==0] = gauss[mask==0]
        mask[:,:,:] = 1
