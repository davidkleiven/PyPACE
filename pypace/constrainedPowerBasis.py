import numpy as np
from scipy import special as spec

class Basis( object ):
    def __init__( self, scaleX, scaleY, scaleZ ):
        self.scaleX = scaleX
        self.scaleY = scaleY
        self.scaleZ = scaleZ

    def phi( self, n, x, scale ):
        return np.sqrt(1.0/scale)*spec.eval_hermite( n, x/scale )*np.exp( -0.5*(x/scale)**2 )

    def phiX( self, n, x ):
        return self.phi( n, x, self.scaleX )

    def phiY( self, n, y ):
        return self.phi( n, y, self.scaleY )

    def phiZ( self, n, z ):
        return self.phi( n, z, self.scaleZ )

    def eval( self, x, y, z, nx, ny, nz ):
        return self.phiX( nx, x )*self.phiY( ny, y )*self.phiZ( nz, z )

    def phiFourier( self, n, k, scale ):
        return 1j*np.sqrt(scale)*spec.eval_hermite( n, k*scale)*np.exp( -0.5*(k*scale)**2 )

    def phiFourierX( self, n, kx ):
        return self.phiFourier( n, kx, self.scaleX )

    def phiFourierY( self, n, ky ):
        return self.phiFourier( n, ky, self.scaleY )

    def phiFourierZ( self, n, kz ):
        return self.phiFourier( n, kz, self.scaleZ )

    def evalFourier( self, kx, ky, kz, nx, ny, nz ):
        return self.phiFourierX( nx, kx )*self.phiFourierY( ny, ky )*self.phiFourierZ( nz, kz )
