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
        self.data = np.zeros((dim,dim,dim))

    #def insert( slice1, angle1Deg, slice2, stepAngleDeg ):


class ProjectionPropagator(object):
    def __init__( self, deltaDistribution, wavelength, voxelsize, kspaceDim=512, maxDeltaValue=5E-5 ):
        self.delta = deltaDistribution*maxDeltaValue/deltaDistribution.max()
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.kspaceDim=kspaceDim

    def yAxisIsMainAxis( self, angle ):
        return angle < np.pi/4.0 or angle > 3.0*np.pi/4.0

    def plotKspace( self, values ):
        centerX = int(values.shape[0]/2)
        plt.imshow( values[centerX,:,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm(), interpolation="none")
        plt.show()

    def generateKspace( self, angleStepDeg ):
        angles = np.linspace( 0, 180, int( 180/angleStepDeg ) )
        angles *= np.pi/180.0

        # Initialize the new kspace matrix
        kspace = np.zeros((self.kspaceDim,self.kspaceDim,self.kspaceDim))

        # Compute the wavenumber
        k = 2.0*np.pi/self.wavelength

        # Compue the projection along the propagation direction
        proj0 = self.delta.sum(axis=2)*self.voxelsize

        # The far field is given by the Fourier Transform of the exit field
        ff0 = np.abs( np.fft.fft2( np.exp(1j*k*proj0)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
        ff0 = np.fft.fftshift(ff0)
        for i in range( 0, len(angles)-1 ):
            print (angles[i]*180.0/np.pi,angles[i+1]*180.0/np.pi)

            # Rotate the material around the x-axis
            self.delta = sciinterp.rotate(self.delta, angleStepDeg, axes=(1,0), reshape=False )

            # Compute the new projection
            proj1 = self.delta.sum(axis=2)*self.voxelsize

            # Far field is given by the new projection
            ff1 = np.abs( np.fft.fft2( np.exp(1j*k*proj1)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
            ff1 = np.fft.fftshift(ff1)
        return kspace
