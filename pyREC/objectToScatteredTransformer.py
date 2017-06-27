import numpy as np
import pyfftw as ftw
import multiprocessing as mp

class Object2ScatteredTransformer( object ):
    def __init__( self, scatteredData ):
        if ( len(scatteredData.shape) != 3 ):
            raise TypeError("The array scatteredData has to have dimensions 3")

        self.objectData = ftw.empty_aligned( scatteredData.shape, dtype="complex128" )
        self.scatteredData = ftw.empty_aligned( scatteredData.shape, dtype="complex128")
        self.fftF = ftw.FFTW(self.objectData,self.scatteredData, direction="FFTW_FORWARD", threads=mp.cpu_count(), axes=(0,1,2))
        self.fftB = ftw.FFTW(self.scatteredData,self.objectData, direction="FFTW_BACKWARD", threads=mp.cpu_count(), axes=(0,1,2))
        self.objectData[:,:,:] = np.zeros(self.scatteredData.shape)
        self.scatteredData[:,:,:] = scatteredData[:,:,:]

    def forward( self ):
        """
        Transforms from object space to the scattered space
        """
        raise NotImplementedError("Child has to implement the member function forward")

    def backward( self ):
        """
        Transforms from scattered space to object space
        """
        raise NotImplementedError("Childs have to implement the member function backward")

class Rytov( Object2ScatteredTransformer ):
    def __init__( self, fourierdata, incidentAmplitude ):
        Object2ScatteredTransformer.__init__( self, fourierdata )
        self.amplitude = incidentAmplitude

    def forward( self ):
        """
        Transforms from object space to scattered space
        """
        self.fftF( normalise_idft=False, ortho=True )

        # Normalize the data
        self.scatteredData[:,:,:] = self.scatteredData/self.amplitude
        self.scatteredData[:,:,:] = self.amplitude*( np.exp(1j*self.scatteredData[:,:,:]) - 1.0 )
        return self.scatteredData

    def backward( self ):
        """
        Transforms from object space to scattered space
        """
        # Convert back to phase
        self.scatteredData[:,:,:] = -1j*self.amplitude*np.log( 1.0 + self.scatteredData[:,:,:]/self.amplitude)
        self.fftB( normalise_idft=False, ortho=True )
        return self.objectData

class FirstBorn( Object2ScatteredTransformer ):
    def __init__( self, kspace ):
        Object2ScatteredTransformer.__init__( self, kspace )

    def forward( self ):
        """
        Transforms the obect space to scattered space
        """
        self.fftF( normalise_idft=False, ortho=True )
        return self.scatteredData

    def backward( self ):
        """
        Transforms back again
        """
        self.fftB( normalise_idft=False, ortho=True )
        return self.objectData
