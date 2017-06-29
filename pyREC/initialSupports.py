import numpy as np

class InitialSupport( object ):
    def __init__( self, shape ):
        self.initial = np.zeros(shape)

class SphericalSupport( InitialSupport ):
    def __init__( self, N, RInPixels, value ):
        InitialSupport.__init__( self, (N,N,N) )
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)
        R = np.sqrt(X**2+Y**2+Z**2)
        self.initial[R<RInPixels] = value

class BoxSupport( InitialSupport ):
    def __init__( self, N, widthInPixels, value ):
        InitialSupport.__init__( self, (N,N,N) )
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)
        mask = np.array( X>-widthInPixels/2.0, dtype=np.uint8) + np.array( X<widthInPixels/2.0, dtype=np.uint8 )
        mask += np.array( Y>-widthInPixels/2.0, dtype=np.uint8) + np.array( Y<widthInPixels/2.0, dtype=np.uint8 )
        mask += np.array( Z>-widthInPixels/2.0, dtype=np.uint8) + np.array( Z<widthInPixels/2.0, dtype=np.uint8 )
        self.initial[mask==6] = value
