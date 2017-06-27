import numpy as np
import pyfftw as ftw
import multiprocessing as mp

class Object2ScatteredTransformer( object ):
    def __init__( self, scatteredData ):
        if ( len(scatteredData.shape) != 3 ):
            raise TypeError("The array scatteredData has to have dimensions 3")
        self.objectData = ftw.empty_aligned( scatteredData.shape, dtype="complex128" )
        self.scatteredData = ftw.empty_aligned( scatteredData.shape, dtype="complex128")
        self.objectData[:,:,:] = np.zeros(self.scatteredData.shape)
        self.scatteredData[:,:,:] = scatteredData[:,:,:]
        self.fftF = ftw.FFTW(self.objectData,self.scatteredData, direction="FFTW_FORWARD", threads=mp.cpu_count())
        self.fftB = ftw.FFTW(self.scatteredData,self.objectData, direction="FFTW_BACKWARD", threads=mp.cpu_count())

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
        super().__init__( fourierdata )
        self.amplitude = incidentAmplitude

    def forward( self ):
        """
        Transforms from object space to scattered space
        """
        self.fftF()
        self.scatteredData[:,:,:] = np.fft.fftshift(self.scatteredData)

        ftNorm = np.sqrt(self.scatteredData.shape[0]*self.scatteredData.shape[1]*self.scatteredData.shape[2])
        # Normalize the data
        self.scatteredData /= (ftNorm*self.amplitude)
        self.scatteredData[:,:,:] = self.amplitude*( np.exp(1j*self.scatteredData[:,:,:]) - 1.0 )
        return self.scatteredData

    def backward( self ):
        """
        Transforms from object space to scattered space
        """
        # Convert back to phase
        self.scatteredData[:,:,:] = -1j*self.amplitude*np.log( 1.0 + self.scatteredData[:,:,:]/self.amplitude)
        self.fftB()
        ftNorm = np.sqrt(self.scatteredData.shape[0]*self.scatteredData.shape[1]*self.scatteredData.shape[2])
        self.objectData[:,:,:] = np.fft.fftshift(self.objectData)
        self.objectData /= ftNorm
        return self.objectData

class FirstBorn( Object2ScatteredTransformer ):
    def __init__( self, realspace ):
        super().__init__( realspace )

    def forward( self ):
        """
        Transforms the obect space to scattered space
        """
        self.fftF()
        self.scatteredData[:,:,:] = np.fft.fftshift( self.scatteredData )
        return self.scatteredData

    def backward( self ):
        """
        Transforms back again
        """
        self.fftB()
        ftNorm = np.sqrt(self.scatteredData.shape[0]*self.scatteredData.shape[1]*self.scatteredData.shape[2])
        self.objectData /= ftNorm
        return self.objectData
