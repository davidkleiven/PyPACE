import numpy as np
import copy
from scipy.ndimage import interpolation as sciinterp
import matplotlib as mpl
from matplotlib import pyplot as plt

class MatrixBuilder3D:
    """
    This class builds a 3D matrix from rotated projections
    It assumes that the data is rotated around the x-axis (axis=0)
    It uses linear interpolation between two neighbouring slices to fill
    one wedge at the time
    """
    def __init__( self, dim ):
        self.dim = dim
        self.data = np.zeros((dim,dim,dim))

    def insert( self, slice1, slice2, angleDeg, stepAngleDeg ):
        assert( slice1.shape[0] == self.dim )
        assert( slice1.shape[1] == self.dim )
        assert( slice2.shape[0] == self.dim )
        assert( slice2.shape[1] == self.dim )
        if ( angleDeg <= 45 ):
            self.insert0to45( slice1, slice2, angleDeg, stepAngleDeg )
        elif ( angleDeg <= 90 ):
            self.insert45to90( slice1, slice2, angleDeg, stepAngleDeg )
        elif ( angleDeg <= 135 ):
            self.insert90to135( slice1, slice2, angleDeg, stepAngleDeg )
        elif ( angleDeg <= 180 ):
            self.insert135to180( slice1, slice2, angleDeg, stepAngleDeg )
        else:
            print ("Specify an angle between 0 and 180")

    def insert0to45( self, slice1, slice2, angleDeg, stepAngleDeg ):
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        for iy in range(0,int(self.dim/2)):
            zmin = int(iy*np.tan(alpha))
            zmax = int(iy*np.tan(alpha+dalpha))+1
            for iz in range(zmin,zmax):
                weight = (iz-zmin)/(zmax-zmin)
                radialIndx = int( self.dim/2 + np.sqrt(iy**2+iz**2) )
                if ( radialIndx < self.dim ):
                    yIndx = int( self.dim/2 + iy )
                    zIndx = int( self.dim/2 + iz )
                    self.data[:,yIndx,zIndx] = slice1[:,radialIndx]*(1.0-weight) + slice2[:,radialIndx]*weight
                radialIndx = int( self.dim/2 - np.sqrt(iy**2+iz**2) )
                if ( radialIndx >= 0 ):
                    yIndx = int( self.dim/2-iy )
                    zIndx = int( self.dim/2-iz )
                    self.data[:,yIndx,zIndx] = slice1[:,radialIndx]*(1.0-weight) + slice2[:,radialIndx]*weight

    def insert45to90( self, slice1, slice2, angleDeg, stepAngleDeg ):
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = np.pi/2.0-alpha-dalpha
        for iz in range(0,int(self.dim/2)):
            ymin = int(iz*np.tan(beta))
            ymax = int(iz*np.tan(beta+dalpha))+1
            for iy in range(ymin,ymax):
                weight = (iy-ymin)/(ymax-ymin)
                radialIndx = int( self.dim/2 + np.sqrt(iy**2+iz**2))
                if ( radialIndx < self.dim ):
                    yIndx = int( self.dim/2 + iy )
                    zIndx = int( self.dim/2 + iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight
                radialIndx = int( self.dim/2 - np.sqrt(iy**2+iz**2) )
                if ( radialIndx >= 0 ):
                    yIndx = int( self.dim/2-iy )
                    zIndx = int( self.dim/2-iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight

    def insert90to135( self, slice1, slice2, angleDeg, stepAngleDeg ):
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = alpha-np.pi/2.0
        for iz in range(0,int(self.dim/2)):
            ymax = int( -iz*np.tan(beta) )
            ymin = int( -iz*np.tan(beta+dalpha) )
            for iy in range(ymin,ymax):
                weight = (iy-ymin)/(ymax-ymin)
                radialIndx = int( self.dim/2 + np.sqrt(iy**2+iz**2))
                if ( radialIndx < self.dim ):
                    yIndx = int( self.dim/2 + iy )
                    zIndx = int( self.dim/2 + iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight
                radialIndx = int( self.dim/2 - np.sqrt(iy**2+iz**2) )
                if ( radialIndx >= 0 ):
                    yIndx = int( self.dim/2-iy )
                    zIndx = int( self.dim/2-iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight

    def insert135to180( self, slice1, slice2, angleDeg, stepAngleDeg ):
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = np.pi-alpha
        for iy in range(0,int(self.dim/2)):
            zmin = int(iy*np.tan(beta-dalpha))
            zmax = int(iy*np.tan(beta))+1
            for iz in range(zmin,zmax):
                weight = (iz-zmin)/(zmax-zmin)
                radialIndx = int( self.dim/2 + np.sqrt(iy**2+iz**2))
                if ( radialIndx < self.dim ):
                    yIndx = int( self.dim/2 - iy )
                    zIndx = int( self.dim/2 + iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight
                radialIndx = int( self.dim/2 - np.sqrt(iy**2+iz**2) )
                if ( radialIndx >= 0 ):
                    yIndx = int( self.dim/2+iy )
                    zIndx = int( self.dim/2-iz )
                    self.data[:,yIndx,zIndx] = slice2[:,radialIndx]*(1.0-weight) + slice1[:,radialIndx]*weight

    def plot( self ):
        centerX = int(self.dim/2)
        plt.imshow( self.data[centerX,:,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm(), interpolation="none")
        plt.show()



class ProjectionPropagator(object):
    def __init__( self, deltaDistribution, wavelength, voxelsize, kspaceDim=512, maxDeltaValue=5E-5 ):
        self.delta = deltaDistribution*maxDeltaValue/deltaDistribution.max()
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.kspaceDim=kspaceDim

    def yAxisIsMainAxis( self, angle ):
        return angle < np.pi/4.0 or angle > 3.0*np.pi/4.0

    def generateKspace( self, angleStepDeg ):
        angles = np.linspace( 0, 180, int( 180/angleStepDeg )+1 )
        angleStepDeg = angles[1]-angles[0]
        #angles *= np.pi/180.0

        # Initialize the new kspace matrix
        kspaceBuilder = MatrixBuilder3D( self.kspaceDim )

        # Compute the wavenumber
        k = 2.0*np.pi/self.wavelength

        # Compue the projection along the propagation direction
        proj0 = self.delta.sum(axis=2)*self.voxelsize

        # The far field is given by the Fourier Transform of the exit field
        ff0 = np.abs( np.fft.fft2( np.exp(1j*k*proj0)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
        ff0 = np.fft.fftshift(ff0)
        for i in range( 0, len(angles)-1 ):
            print (angles[i],angles[i+1])

            # Rotate the material around the x-axis
            self.delta = sciinterp.rotate(self.delta, angleStepDeg, axes=(1,0), reshape=False )

            # Compute the new projection
            proj1 = self.delta.sum(axis=2)*self.voxelsize

            # Far field is given by the new projection
            ff1 = np.abs( np.fft.fft2( np.exp(1j*k*proj1)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
            ff1 = np.fft.fftshift(ff1)
            kspaceBuilder.insert( ff0, ff1, angles[i], angleStepDeg )
            ff0[:,:] = ff1[:,:]
            kspaceBuilder.plot()
        return kspaceBuilder.data
