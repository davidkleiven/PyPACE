import numpy as np
import pyfftw as ftw
import multiprocessing as mp
from scipy import ndimage

class Object2ScatteredTransformer( object ):
    def __init__( self, scatteredData, numpyFFT=False ):
        if ( len(scatteredData.shape) != 3 ):
            raise TypeError("The array scatteredData has to have dimensions 3")

        if ( numpyFFT ):
            self.objectData = np.zeros( scatteredData.shape, dtype=np.complex128 )
            self.scatteredData = np.zeros( scatteredData.shape, dtype=np.complex128 )
        else:
            self.objectData = ftw.empty_aligned( scatteredData.shape, dtype="complex128" )
            self.scatteredData = ftw.empty_aligned( scatteredData.shape, dtype="complex128")
            self.fftF = ftw.builders.fftn(self.objectData, threads=mp.cpu_count() )
            self.fftB = ftw.builders.ifftn(self.scatteredData, threads=mp.cpu_count() )
        self.objectData[:,:,:] = np.zeros(self.scatteredData.shape)
        self.scatteredData[:,:,:] = scatteredData[:,:,:]
        self.numpyFFT = numpyFFT

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
        self.objectData [:,:,:] = self.fftB( normalise_idft=False, ortho=True )
        return self.objectData

class FirstBorn( Object2ScatteredTransformer ):
    def __init__( self, kspace, numpyFFT=False ):
        Object2ScatteredTransformer.__init__( self, kspace, numpyFFT=numpyFFT )

    def forward( self ):
        """
        Transforms the object space to scattered space
        """
        if ( self.numpyFFT ):
            self.scatteredData = np.fft.fftn( self.objectData, norm="ortho" )
        else:
            self.scatteredData[:,:,:] = self.fftF( normalise_idft=False, ortho=True )
        return self.scatteredData

    def backward( self ):
        """
        Transforms back again
        """
        if ( self.numpyFFT ):
            self.objectData = np.fft.ifftn( self.scatteredData, norm="ortho" )
        else:
            self.objectData[:,:,:] = self.fftB( normalise_idft=False, ortho=True )
        return self.objectData
