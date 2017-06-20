import numpy as np
import copy
from scipy.ndimage import interpolation as sciinterp

class ProjectionPropagator(object):
    def __init__( self, deltaDistribution, wavelength, voxelsize, kspaceDim=512, maxDeltaValue=5E-5 ):
        self.delta = deltaDistribution*maxDeltaValue/deltaDistribution.max()
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.kspaceDim=kspaceDim

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
            for iy in range( int(self.kspaceDim/2) ):
                zmin = int( -iy*np.sin(angles[i+1]) )
                zmax = int( -iy*np.sin(angles[i]) )
                for iz in range(zmin,zmax):
                    weight = (iz-zmin)/(zmax-zmin)
                    zIndx = int( self.kspaceDim/2+iz )
                    yIndx = int( self.kspaceDim/2+iy)
                    indx = int( np.sqrt( iz**2 + iy**2) + self.kspaceDim/2 )
                    if ( indx >= 512 ):
                        break
                    kspace[:,yIndx,zIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
                zmin = int( iy*np.sin(angles[i]) )
                zmax = int( iy*np.sin(angles[i+1]))

                for iz in range(zmin,zmax):
                    weight = (iz-zmin)/(zmax-zmin)
                    zIndx = int( self.kspaceDim/2+iz )
                    yIndx = int( self.kspaceDim/2-iy)
                    indx = int( self.kspaceDim/2 - np.sqrt( iz**2 + iy**2) )
                    if ( indx < 0 ):
                        break
                    kspace[:,yIndx,zIndx] = ff1[:,indx]*weight + (1.0-weight)*ff0[:,indx]
            ff0[:,:] = ff1[:,:]
        return kspace
