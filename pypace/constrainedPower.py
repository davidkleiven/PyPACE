import constrainedPowerBasis as cpb
import numpy as np
from matplotlib import pyplot as plt

class ConstrainedPower( object ):
    def __init__( self, mask, support, Nbasis=3 ):
        self.mask = mask
        self.support = support
        self.scaleX, self.scaleY, self.scaleZ = self.computeScales()
        self.basis = cpb.Basis( self.scaleX, self.scaleY, self.scaleZ )
        self.Nbasis = Nbasis
        self.eigval = None
        self.eigvec = None

    def computeScales( self ):
        N = self.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)

        sumD = self.support.sum()
        scaleX = np.sqrt( np.sum( X**2 *self.support)/sumD )
        scaleY = np.sqrt( np.sum( Y**2 *self.support)/sumD )
        scaleZ = np.sqrt( np.sum( Z**2 *self.support)/sumD )
        return scaleX, scaleY, scaleZ

    def flattenedToXYZ( self, flattened ):
        iz = flattened%self.Nbasis
        iy = int( flattened/self.Nbasis )
        ix = int( flattened/self.Nbasis**2 )
        return ix,iy,iz

    def operatorElement( self, n, m ):
        ix1, iy1, iz1 = self.flattenedToXYZ(n)
        ix2, iy2, iz2 = self.flattenedToXYZ(m)
        N = self.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)
        outsideSupport = self.mask.shape[0]**3 - np.count_nonzero(self.mask)

        # Contribution from outside the support
        func1 = self.basis.eval( X,Y,Z, ix1, iy1, iz1 )
        func2 = self.basis.eval( X,Y,Z, ix2, iy2, iz2 )
        norm1 = np.sqrt( np.sum(np.abs(func1)**2) )
        norm2 = np.sqrt( np.sum(np.abs(func2)**2) )
        realSp = np.sum( np.conj(func1[self.support==0])*func2[self.support==0] )/(norm1*norm2)

        # Contribution from Fourier domain
        del X,Y,Z,func1,func2
        
        N = self.mask.shape[0]
        kx = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        ky = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        kz = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        KX,KY,KZ = np.meshgrid(kx,ky,kz)
        func1 = self.basis.evalFourier( KX, KY, KZ, ix1, iy1, iz1 )
        func2 = self.basis.evalFourier( KX, KY, KZ, ix2, iy2, iz2 )
        norm1 = np.sqrt( np.sum( np.abs(func1)**2 ) )
        norm2 = np.sqrt( np.sum( np.abs(func2)**2 ) )

        insideMask = np.count_nonzero(self.mask)
        fourierSpace = np.sum( np.conj(func1[self.mask==1])*func2[self.mask==1] )/(norm1*norm2)
        return realSp + fourierSpace

    def buildMatrix( self ):
        mat = np.zeros( (self.Nbasis**3,self.Nbasis**3) ) + 1j*np.zeros( (self.Nbasis**3,self.Nbasis**3) )
        for i in range(self.Nbasis**3):
            print ("Computing row %d of %d"%(i,self.Nbasis**3))
            for j in range(i,self.Nbasis**3):
                mat[i,j] = self.operatorElement(i,j)
                mat[j,i] = np.conj(mat[i,j])
        return mat

    def plotEigenvalues( self ):
        if ( self.eigval is None ):
            return None
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.eigval)
        return fig

    def solve( self ):
        mat = self.buildMatrix()
        plt.matshow(np.abs(mat))
        plt.colorbar()
        plt.show()
        self.eigval, self.eigvec = np.linalg.eigh(mat)
        return self.eigval, self.eigvec
