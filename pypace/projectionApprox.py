import numpy as np
import copy
from scipy.ndimage import interpolation as sciinterp

class ProjectionPropagator(object):
    def __init__( self, deltaDistribution, wavelength, voxelsize, kspaceDim=512, maxDeltaValue=5E-5 ):
        self.delta = deltaDistribution*maxDeltaValue/deltaDistribution.max()
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.kspaceDim=kspaceDim

    def yAxisIsMainAxis( self, angle ):
        return angle < 45 or angle > 135

    def generateKspace( self, angleStepDeg ):
        angles = np.linspace( 0, 45, int( 45/angleStepDeg ) )
        angles *= np.pi/180.0
        kspace = np.zeros((self.kspaceDim,self.kspaceDim,self.kspaceDim))
        k = 2.0*np.pi/self.wavelength
        proj0 = self.delta.sum(axis=2)*self.voxelsize
        ff0 = np.abs( np.fft.fft2( np.exp(1j*k*proj0)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
        ff0 = np.fft.fftshift(ff0)
        for i in range( 0, len(angles)-1 ):
            print (angles[i])
            self.delta = sciinterp.rotate(self.delta, angleStepDeg, axes=(1,0), reshape=False )
            proj1 = self.delta.sum(axis=2)*self.voxelsize
            ff1 = np.abs( np.fft.fft2( np.exp(1j*k*proj1)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
            ff1 = np.fft.fftshift(ff1)
            for imain in range( int(self.kspaceDim/2) ):
                if ( self.yAxisIsMainAxis(angles[i]) ):
                    secondmin = int( -imain*np.sin(angles[i+1]) )
                    secondmax = int( -imain*np.sin(angles[i]) )
                else:
                    secondmin = int( -imain*np.sin(np.pi/2.0-angles[i]) )
                    secondmax = int( -imain*np.sin(np.pi/2.0-angles[i+1]) )

                for isecond in range(secondmin,secondmax):
                    weight = (isecond-secondmin)/(secondmax-secondmin)
                    secondIndx = int( self.kspaceDim/2+isecond )
                    mainIndx = int( self.kspaceDim/2+imain)
                    indx = int( np.sqrt( isecond**2 + imain**2) + self.kspaceDim/2 )
                    if ( indx >= 512 ):
                        break
                    if ( self.yAxisIsMainAxis(angles[i]) ):
                        kspace[:,mainIndx,secondIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
                    else:
                        kspace[:,secondIndx,mainIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
                if ( self.yAxisIsMainAxis(angles[i]) ):
                    secondmin = int( imain*np.sin(angles[i]) )
                    secondmax = int( imain*np.sin(angles[i+1]))
                else:
                    secondmin = int( imain*np.sin(np.pi/2.0-angles[i+1]) )
                    secondmax = int( imain*np.sin(np.pi/2.0-angles[i]))

                for isecond in range(secondmin,secondmax):
                    weight = (isecond-secondmin)/(secondmax-secondmin)
                    secondIndx = int( self.kspaceDim/2+isecond )
                    mainIndx = int( self.kspaceDim/2-imain)
                    indx = int( self.kspaceDim/2 - np.sqrt( isecond**2 + imain**2) )
                    if ( indx < 0 ):
                        break
                    if ( self.yAxisIsMainAxis(angles[i]) ):
                        kspace[:,mainIndx,secondIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
                    else:
                        kspace[:,secondIndx,mainIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
            ff0[:,:] = ff1[:,:]
        return kspace
