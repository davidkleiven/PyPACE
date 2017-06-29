import numpy as np
import objectToScatteredTransformer as otst
import constraints as cnst
import matplotlib as mpl
import config
if ( not config.enableMPLShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
from scipy import ndimage
import supports as sup
import initialSupports as isup
import cytParallel as cytp

class Reconstructor( object ):
    def __init__( self, obj2ScatTrans, fractionOfMeanThreshold, beta=0.9, maxIter=500, statusEvery=10,
    nIterAtEndWithFixed=60 ):
        if ( not isinstance(obj2ScatTrans, otst.Object2ScatteredTransformer ) ):
            raise TypeError("Obj2ScatTrans has to of type Object2ScatteredTransformer ")
        if ( fractionOfMeanThreshold < 0.0 or fractionOfMeanThreshold > 1.0 ):
            raise ValueError("fractionOfMeanThreshold has to be in the range (0,1)")

        self.transformer = obj2ScatTrans

        # Define the supports to be used
        self.mask = np.zeros(self.transformer.objectData.shape, dtype=np.uint8 )
        self.support = sup.Support( self.mask, 0.05 )

        self.lastObject = np.zeros(self.transformer.objectData.shape)+1j*np.zeros(self.transformer.objectData.shape)

        # Define the different update algorithms to be used
        self.signflip = cnst.SignFlip( self.support )
        self.hybrid = cnst.Hybrid( beta, self.lastObject, self.support )
        self.fixed = cnst.FixedSupport( self.support )
        self.fourier = cnst.FourierConstraint( self.transformer.scatteredData )

        self.residuals = np.zeros(maxIter+nIterAtEndWithFixed)
        self.currentIter = 0
        self.statusEvery = statusEvery
        self.maxIter = maxIter
        self.nIterAtEndWithFixed = nIterAtEndWithFixed
        self.bestState = np.zeros((self.transformer.objectData.shape))
        self.minResidual = 1E30
        self.phasesAreInitialized = False
        self.supportInitialized = False
        self.isFirstIteration = True
        self.updateSupportEvery = 2000

    def initScatteredDataWithRandomPhase( self ):
        shape = self.transformer.scatteredData.shape
        self.transformer.scatteredData *= np.exp( 1j*np.random.rand(shape[0],shape[1],shape[2])*2.0*np.pi)
        self.phasesAreInitialized = True

    def initDataWithKnownSupport( self, support ):
        if ( not isinstance( support, isup.InitialSupport) ):
            raise TypeError("The support argument passed to initDataWithKnownSupport has to by of type InitialSupport")
        self.supportInitialized = True
        if ( support.initial.shape != self.transformer.objectData.shape ):
            raise TypeError("The shape of the initial support does not match the shape of the scattered data")

        self.transformer.objectData[:,:,:] = support.initial

    def printStatus( self ):
            print ("Iteration: %d, residual %.2E"%(self.currentIter, self.residuals[self.currentIter]))

    def getFourierResidual( self ):
        #res = np.sqrt( np.sum( np.abs(self.fourier.measured - np.abs(ed error in the object domain can be eself.transformer.scatteredData))**2 ) )
        mod = np.empty(self.transformer.scatteredData.shape)
        cytp.modulus( self.transformer.scatteredData, mod )
        return cytp.meanSquareError( self.fourier.measured, mod )

    def step( self, constraint ):
        if ( not self.phasesAreInitialized and not self.supportInitialized ):
            raise RuntimeError("Phases or support needs to be initialized")

        if ( self.phasesAreInitialized or self.currentIter > 0 ):
            # If the user has initialized the phase, the first step should
            # be to go from the scattered domain to object domain
            self.transformer.backward()
            constraint.apply( self.transformer.objectData )
        self.transformer.forward()

        self.residuals[self.currentIter] = self.getFourierResidual()
        if ( self.residuals[self.currentIter] < self.minResidual):
            cytp.modulus( self.transformer.objectData, self.bestState )
            self.minResidual = self.residuals[self.currentIter]

        self.fourier.apply( self.transformer.scatteredData )

        # Copy the current object to the last object array
        cytp.copy( self.transformer.objectData, self.lastObject )

        if ( self.currentIter%self.statusEvery == 0 ):
            self.printStatus()
        if ( self.currentIter%self.updateSupportEvery == 0 ):
            self.support.update( self.transformer.objectData )
        self.currentIter += 1

    def run( self, graphicUpdate=False ):
        #for i in range(0,60):
        #    self.step( self.signflip )

        finished = False
        if ( graphicUpdate ):
            plt.ion()
            fig = plt.figure()
        while ( not finished ):

            for i in range(0,10):
                self.step( self.hybrid )
                if ( graphicUpdate ):
                    self.plotCurrent(fig=fig)
                    plt.draw()
                    plt.pause(0.01)
                if ( self.currentIter >= self.maxIter ):
                    finished=True
                    break
            if ( finished ):
                break
            """
            for i in range(0,10):
                self.step( self.signflip)
                if ( graphicUpdate ):
                    self.plotCurrent(fig=fig)
                    plt.draw()
                    plt.pause(0.01)
                if ( self.currentIter >= self.maxIter ):
                    finished=True
                    break
            if ( finished ):
                break
            """

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
        return self.plot2DSlices( self.bestState )

    def plotCurrent( self, fig=None ):
        return self.plot2DSlices( np.abs(self.transformer.objectData), fig=fig)

    def plot2DSlices( self, data, fig=None ):
        if ( fig is None ):
            fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        com = ndimage.measurements.center_of_mass(data)
        ax1.imshow( data[int(com[0]),:,:], cmap="bone", interpolation="none")
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow( data[:,int(com[1]),:], cmap="bone", interpolation="none")

        ax3 = fig.add_subplot(1,3,3)
        im = ax3.imshow( data[:,:,int(com[2])], cmap="bone", interpolation="none")
        #fig.colorbar(im)
        return fig

    def plotFourierSlices( self ):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        com = ndimage.measurements.center_of_mass(np.abs(self.transformer.scatteredData))
        im = ax1.imshow( np.abs(self.transformer.scatteredData[0,:,:]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())
        fig.colorbar(im)
        ax2 = fig.add_subplot(1,3,2)
        im = ax2.imshow( np.abs(self.transformer.scatteredData[:,0,:]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())
        fig.colorbar(im)

        ax3 = fig.add_subplot(1,3,3)
        im = ax3.imshow( np.abs(self.transformer.scatteredData[:,:,0]), cmap="inferno", interpolation="none", norm=mpl.colors.LogNorm())
        fig.colorbar(im)

    def save( self, fname="reconstruction.raw" ):
        # Convert to uint8
        data = np.abs( self.transformer.objectData )
        maxval = np.max( data )
        data *= (255.0/maxval)
        data = data.astype(np.uint8)
        data = data.T
        data.tofile( fname )
        print ("Data stored to %s"%(fname))
