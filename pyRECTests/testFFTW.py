import unittest
import numpy as np
from matplotlib import pyplot as plt
import pyfftw as ftw

class TestFFTW( unittest.TestCase ):
    def testFFTW( self ):
        N = 128
        x = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,x,x)
        R = np.sqrt( X**2 + Y**2 + Z**2 )
        del X,Y,Z

        values = np.zeros((N,N,N))
        values[R<N/4] = 1.0
        ff = np.fft.fftn( values, norm="ortho" )
        ff = np.fft.fftshift(ff)
        ffAmp = np.abs(ff)**2

        val = ftw.empty_aligned( (N,N,N), dtype="complex128" )
        val[:,:,:] = values
        ft = ftw.builders.fftn( val, threads=2 )
        ffw = ft( normalise_idft=False, ortho=True )
        ffw = np.fft.fftshift(ffw)
        ffwAmp = np.abs(ffw)**2
        diff = np.sum( (ffAmp-ffwAmp)**2 )
        self.assertAlmostEqual( diff, 0.0, places=6 )

        # Check the inverse transforms
        npInv = np.abs( np.fft.ifftn( ff, norm="ortho" ) )
        diffNpy = np.sum( (npInv-values)**2 )
        self.assertAlmostEqual( diffNpy, 0.0, places=6 )

        iffw = ftw.builders.ifftn( ffw, threads=2 )
        invFFW = np.abs( iffw( normalise_idft=False, ortho=True ) )

        diffFFTW = np.sum( (invFFW-values)**2 )
        self.assertAlmostEqual( diffFFTW, 0.0, places=6 )

if __name__ == "__main__":
    unittest.main()
