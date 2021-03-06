from __future__ import print_function, division
import config
import numpy as np
import copy
from scipy.ndimage import interpolation as sciinterp
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt
from mpi4py import MPI

class MatrixBuilder3D:
    """
    This class builds a 3D matrix from rotated projections
    It assumes that the data is rotated around the x-axis (axis=0)
    It uses linear interpolation between two neighbouring slices to fill
    one wedge at the time

    dim: int
        Number of dimensions of one of the directon of the matrix.
        The matrix is assumed to have shape [dim,dim,dim]
    """
    def __init__( self, dim ):
        self.dim = dim
        self.data = np.zeros((dim,dim,dim))

    def insert( self, slice1, slice2, angleDeg, stepAngleDeg ):
        """
        Inserts slices into the 3D matrix

        slice1: ndarray
            2D array containing the first slice in K-space

        slice2: ndarray
            2D array containing the second slicei in K-space

        angleDeg: float
            Angle of the first slice

        stepAngleDeg: float
            Difference between the angle of the first slice and the second slice
        """
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
        #self.plot()

    def insert0to45( self, slice1, slice2, angleDeg, stepAngleDeg ):
        """
        Insert slices when the angle of rotation alpha satisfy 0 < alpha <= 45

        See: :func:'~projectionApprox.MatrixBuilder3D.insert'
        """
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        for iy in range(0,int(self.dim/2)):
            zmin = int(iy*np.tan(alpha))
            zmax = int(iy*np.tan(alpha+dalpha))+1
            for iz in range(zmin,zmax):
                if ( iy == 0 ):
                    angle = alpha
                else:
                    angle = np.arctan(iz/iy)

                #weight = (iz-zmin)/(zmax-zmin)
                weight = (angle-alpha)/dalpha
                if ( weight > 1.0 ):
                    weight = 1.0
                elif ( weight < 0.0 ):
                    weight = 0.0
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
        """
        Insert slices when the angle of rotation alpha satisfy 45 < alpha <= 90

        See: :func:'~projectionApprox.MatrixBuilder3D.insert'
        """
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = np.pi/2.0-alpha-dalpha
        for iz in range(0,int(self.dim/2)):
            ymin = int(iz*np.tan(beta))
            ymax = int(iz*np.tan(beta+dalpha))+1
            for iy in range(ymin,ymax):
                if ( iz == 0 ):
                    angle = beta
                else:
                    angle = np.arctan(iy/iz)
                weight = (angle-beta)/dalpha
                if ( weight > 1.0 ):
                    weight = 1.0
                elif ( weight < 0.0 ):
                    weight = 0.0
                assert( weight >= 0.0 and weight <= 1.0 )
                #weight = (iy-ymin)/(ymax-ymin)
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
        """
        Insert slices when the angle of rotation alpha satisfy 90 < alpha <= 135

        See: :func:'~projectionApprox.MatrixBuilder3D.insert'
        """
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = alpha-np.pi/2.0
        for iz in range(0,int(self.dim/2)):
            ymax = int( -iz*np.tan(beta) )
            ymin = int( -iz*np.tan(beta+dalpha) )
            for iy in range(ymin,ymax):
                if ( iz == 0 ):
                    angle = beta+dalpha
                else:
                    angle = np.arctan(-iy/iz)
                weight = ( beta+dalpha-angle )/dalpha
                if ( weight > 1.0 ):
                    weight = 1.0
                elif ( weight < 0.0 ):
                    weight = 0.0
                #weight = (iy-ymin)/(ymax-ymin)
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
        """
        Insert slices when the angle of rotation alpha satisfy135 < alpha <= 180

        See: :func:'~projectionApprox.MatrixBuilder3D.insert'
        """
        alpha = angleDeg*np.pi/180.0
        dalpha = stepAngleDeg*np.pi/180.0
        beta = np.pi-alpha
        for iy in range(0,int(self.dim/2)):
            zmin = int(iy*np.tan(beta-dalpha))
            zmax = int(iy*np.tan(beta))+1
            for iz in range(zmin,zmax):
                if ( iy == 0 ):
                    angle = beta-dalpha
                else:
                    angle = np.arctan(iz/iy)
                weight = ( angle - beta+dalpha )/dalpha
                if ( weight > 1.0 ):
                    weight = 1.0
                elif ( weight < 0.0 ):
                    weight = 0.0
                #weight = (iz-zmin)/(zmax-zmin)
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
        """
        Plot a slice in the yz-plane of the 3D data
        """
        centerX = int(self.dim/2)
        plt.imshow( self.data[centerX,:,:], cmap="nipy_spectral", norm=mpl.colors.LogNorm(), interpolation="none")
        plt.show()



class ProjectionPropagator(object):
    """
    Computes the exit field after propagating across an object using the projection approximation

    deltaDistribution: ndarray
        3D array containing the deviation from unitty of the real part of the refractive index

    wavelength: float
        Wavelength of the X-rays in nano meters

    voxelsize: float
        Voxelsize in nano meters

    kspaceDim: int
        Number of elements along one direction of the k-space matrix. The k-space matrix is assumed to have shape
        [kspaceDim,kspaceDim,kspaceDim]

    maxDeltaValue: float
        Maximum allowed value for the deviation from unity of the real part of the refractive index
    """
    def __init__( self, deltaDistribution, wavelength, voxelsize, kspaceDim=512, maxDeltaValue=5E-5 ):
        self.delta = deltaDistribution*maxDeltaValue/deltaDistribution.max()
        self.deltaBackup = copy.deepcopy(self.delta)
        self.wavelength = wavelength
        self.voxelsize = voxelsize
        self.kspaceDim=kspaceDim

    def getFarField( self ):
        """
        Computes the far field scattering pattern based on the projection approxmiation
        """
        k = 2.0*np.pi/self.wavelength
        proj0 = self.delta.sum(axis=2)*self.voxelsize
        ff0 = np.abs( np.fft.fft2( np.exp(1j*k*proj0)-1.0, s=(self.kspaceDim,self.kspaceDim) ) )**2
        ff0 = np.fft.fftshift(ff0)
        return ff0

    def generateKspace( self, angleStepDeg ):
        """
        Generates full 3D kspace representation from the projection approximation

        angleStepDeg: float
            Rotation step in degree between successive rotations

        Returns: ndarray
            3D array representing the Fourier space
        """
        angles = np.linspace( 0, 180, int( 180/angleStepDeg )+1 )
        angleStepDeg = angles[1]-angles[0]
        #angles *= np.pi/180.0

        # Initialize the new kspace matrix
        kspaceBuilder = MatrixBuilder3D( self.kspaceDim )

        ff0 = self.getFarField()
        baseAngle = 0
        rotDir = 1
        baseIter = 0
        for i in range( 0, len(angles)-1 ):
            #print (angles[i],angles[i+1])

            # Rotate the material around the x-axis
            self.delta = sciinterp.rotate(self.deltaBackup, angles[i+1], axes=(1,0), reshape=False )

            ff1 = self.getFarField()
            kspaceBuilder.insert( ff0, ff1, angles[i], angleStepDeg )
            ff0[:,:] = ff1[:,:]
            #kspaceBuilder.plot()
        return kspaceBuilder.data
