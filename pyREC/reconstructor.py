import numpy as np
import objectToScatteredTransformer as otst
import constraints as cnst
import matplotlib as mpl
import config
if ( not config.enableMPLShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
from scipy import ndimage

class Reconstructor( object ):
    def __init__( self, obj2ScatTrans, fractionOfMeanThreshold, beta=0.9, maxIter=500, statusEvery=10,
    nIterAtEndWithFixed=60 ):
        if ( not isinstance(obj2ScatTrans, otst.Object2ScatteredTransformer ) ):
            raise TypeError("Obj2ScatTrans has to of type Object2ScatteredTransformer ")
        if ( fractionOfMeanThreshold < 0.0 or fractionOfMeanThreshold > 1.0 ):
            raise ValueError("fractionOfMeanThreshold has to be in the range (0,1)")

        self.transformer = obj2ScatTrans
        self.lastObject = np.zeros(self.transformer.objectData.shape)+1j*np.zeros(self.transformer.objectData.shape)
        self.signflip = cnst.SignFlip( fractionOfMeanThreshold )
        self.hybrid = cnst.Hybrid( fractionOfMeanThreshold, beta, self.lastObject )
        self.fixed = cnst.FixedSupport( fractionOfMeanThreshold )
        self.fourier = cnst.FourierConstraint( self.transformer.scatteredData )
        #self.initScatteredDataWithRandomPhase()
        self.residuals = np.zeros(maxIter+nIterAtEndWithFixed)
        self.currentIter = 0
        self.statusEvery = statusEvery
        self.maxIter = maxIter
        self.nIterAtEndWithFixed = nIterAtEndWithFixed
        self.bestState = np.zeros((self.transformer.objectData.shape))
        self.minResidual = 1E30

    def initScatteredDataWithRandomPhase( self ):
        shape = self.transformer.scatteredData.shape
        self.transformer.scatteredData *= np.exp( 1j*np.random.rand(shape[0],shape[1],shape[2])*2.0*np.pi)

    def printStatus( self ):
            print ("Iteration: %d, residual %.2E"%(self.currentIter, self.residuals[self.currentIter]))

    def getFourierResidual( self ):
        res = np.sqrt( np.sum( np.abs(self.fourier.measured - np.abs(self.transformer.scatteredData))**2 ) )
        shape = self.fourier.measured.shape
        res /= np.sum(self.fourier.measured)
        return res

    def step( self, constraint ):
        self.transformer.backward()
        constraint.apply( self.transformer.objectData )
        self.transformer.forward()
        self.residuals[self.currentIter] = self.getFourierResidual()
        if ( self.residuals[self.currentIter] < self.minResidual):
            self.bestState[:,:,:] = np.abs(self.transformer.objectData)
            self.minResidual = self.residuals[self.currentIter]
        self.transformer.scatteredData[:,:,:] = self.fourier.apply( self.transformer.scatteredData )

        # Copy the current object to the last object array
        self.lastObject[:,:,:] = self.transformer.objectData[:,:,:]

        if ( self.currentIter%self.statusEvery == 0 ):
            self.printStatus()
        self.currentIter += 1

    def run( self ):
        for i in range(0,60):
            self.step( self.signflip )

        finished = False
        while ( not finished ):
            for i in range(0,10):
                self.step( self.hybrid )
                if ( self.currentIter >= self.maxIter ):
                    finished=True
                    break
            if ( finished ):
                break

            for i in range(0,10):
                self.step( self.signflip )
                if ( self.currentIter >= self.maxIter ):
                    finished=True
                    break
            if ( finished ):
                break

        for i in range(0,self.nIterAtEndWithFixed):
            self.step( self.fixed )


    def plotResidual( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.residuals )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fourier Residual")
        ax.set_yscale("log")
        fname = "residual.png"
        fig.savefig(fname)
        print ("Residual saved to %s"%(fname))


    def plotBest( self ):
        self.plot2DSlices( self.bestState )

    def plotCurrent( self ):
        self.plot2DSlices( np.abs(self.transformer.objectData) )

    def plot2DSlices( self, data ):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        com = ndimage.measurements.center_of_mass(data)
        ax1.imshow( data[int(com[0]),:,:], cmap="inferno", interpolation="none")
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( data[:,int(com[1]),:], cmap="inferno", interpolation="none")

        ax3 = fig.add_subplot(1,3,3)
        im = ax3.imshow( data[:,:,int(com[2])], cmap="inferno", interpolation="none")
        fig.colorbar(im)

    def plotFourierSlices( self ):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        com = ndimage.measurements.center_of_mass(np.abs(self.transformer.scatteredData))
        ax1.imshow( np.abs(self.transformer.scatteredData[int(com[0]),:,:]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( np.abs(self.transformer.scatteredData[:,int(com[1]),:]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())

        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow( np.abs(self.transformer.scatteredData[:,:,int(com[2])]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())
