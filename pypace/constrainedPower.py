import sys
sys.path.append("./")
import constrainedPowerBasis as cpb
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as interp
import constrainedpowerc as cnstpow

class ConstrainedPower( object ):
    def __init__( self, mask, support, Nbasis=3 ):
        self.mask = mask
        self.support = support
        self.scaleX, self.scaleY, self.scaleZ = self.computeScales()
        self.basis = cpb.Basis( self.scaleX, self.scaleY, self.scaleZ )
        self.Nbasis = Nbasis
        self.eigval = None
        self.eigvec = None
        N = self.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        self.supInterp = interp.RegularGridInterpolator( (x,x,x), self.support, fill_value=0, bounds_error=False )

        N = self.mask.shape[0]
        kx = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        self.maskInterp = interp.RegularGridInterpolator( (kx,kx,kx), self.mask, fill_value=0, bounds_error=False )
        self.integrationOrder = 20
        self.points, self.weights = np.polynomial.hermite.hermgauss( self.integrationOrder )

    def computeScales( self ):
        N = self.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)

        sumD = self.support.sum()
        scaleX = np.sqrt( np.sum( X**2 *self.support)/sumD )/2.0
        scaleY = np.sqrt( np.sum( Y**2 *self.support)/sumD )/2.0
        scaleZ = np.sqrt( np.sum( Z**2 *self.support)/sumD )/2.0
        return scaleX, scaleY, scaleZ

    def flattenedToXYZ( self, flattened ):
        iz = flattened%self.Nbasis
        iy = ( int( flattened/self.Nbasis )%self.Nbasis )
        ix = int( flattened/self.Nbasis**2 )
        return ix,iy,iz

    def integrate( self, n, m, weight="none" ):
        nx1, ny1, nz1 = self.flattenedToXYZ(n)
        nx2, ny2, nz2 = self.flattenedToXYZ(m)
        integral = 0.0
        x = self.basis.scaleX*self.points
        y = self.basis.scaleY*self.points

        for ix in range(len(x)):
            for iy in range(len(y)):
                z = self.points*self.basis.scaleZ
                if ( weight=="none" ):
                    supportContrib = np.ones(len(self.weights))
                elif ( weight=="support" ):
                    xpt = np.ones(len(z))*x[ix]#/self.basis.scaleX
                    ypt = np.ones(len(x))*y[iy]#/self.basis.scaleY
                    pts = np.vstack((xpt,ypt,z)).T#/self.basis.scaleZ)).T
                    supportContrib = 1.0-self.supInterp( (xpt,ypt,z) )
                else:
                    raise ValueError("Weight has to none or support")
                    # Integrate basis function n multiplied with basis function m
                integral += np.sum( np.conj( self.basis.evalNoWeight(x[ix],y[iy],z,nx1,ny1,nz1) )*
                self.basis.evalNoWeight(x[ix],y[iy],z,nx2,ny2,nz2)*supportContrib*
                self.weights )*self.weights[iy]*self.weights[ix]
        return integral

    def integrateFourier( self, n, m, weight="none" ):
        nx1, ny1, nz1 = self.flattenedToXYZ(n)
        nx2, ny2, nz2 = self.flattenedToXYZ(m)
        integral = 0.0
        kx = self.points/self.basis.scaleX
        ky = self.points/self.basis.scaleY

        for ix in range(len(kx)):
            for iy in range(len(ky)):
                kz = self.points/self.basis.scaleZ
                if ( weight == "none" ):
                    maskContrib = np.ones(len(self.weights))
                elif ( weight == "mask" ):
                    kxpts = np.ones(len(kz))*kx[ix]#*self.basis.scaleX
                    kypts = np.ones(len(kz))*ky[iy]#*self.basis.scaleY
                    #pts = np.vstack( (kxpts,kypts,kz) ).T#*self.basis.scaleZ) ).T
                    #maskContrib = self.maskInterp( pts )
                    maskContrib = self.maskInterp( (kxpts,kypts,kz) )
                else:
                    raise ValueError("weight has to be either none or mask" )
                integral += np.sum( np.conj(self.basis.evalFourier(kx[ix],ky[iy],kz,nx1,ny1,nz1))*
                self.basis.evalFourier(kx[ix],ky[iy],kz,nx2,ny2,nz2)*maskContrib*self.weights )*self.weights[ix]*self.weights[iy]
        return integral

    def checkOrthogonality( self ):
        for i in range(0,self.Nbasis**3):
            for j in range(0,self.Nbasis**3):
                ix1, iy1, iz1 = self.flattenedToXYZ(i)
                ix2, iy2, iz2 = self.flattenedToXYZ(j)
                norm1 = np.sqrt( self.integrate(i,i,weight="none") )
                norm2 = np.sqrt( self.integrate(j,j,weight="none") )
                #innerprod = np.sum( np.conj(func1)*func2 )/(norm1*norm2)
                innerprod = self.integrate(i,j,weight="none")/( norm1*norm2 )
                print ("(",ix1,iy1,iz1,"), (", ix2,iy2,iz2,") ",innerprod)

    def operatorElement( self, n, m ):
        ix1, iy1, iz1 = self.flattenedToXYZ(n)
        ix2, iy2, iz2 = self.flattenedToXYZ(m)
        #N = self.support.shape[0]
        #x = np.linspace( -N/2, N/2, N )
        #y = np.linspace( -N/2, N/2, N )
        #z = np.linspace( -N/2, N/2, N )
        #X,Y,Z = np.meshgrid(x,y,z)
        #outsideSupport = self.mask.shape[0]**3 - np.count_nonzero(self.mask)

        # Contribution from outside the support
        #func1 = self.basis.eval( X,Y,Z, ix1, iy1, iz1 )
        #func2 = self.basis.eval( X,Y,Z, ix2, iy2, iz2 )
        #norm1 = np.sqrt( np.sum(np.abs(func1)**2) )
        #norm2 = np.sqrt( np.sum(np.abs(func2)**2) )
        #realSp = np.sum( np.conj(func1[self.support==0])*func2[self.support==0] )/(norm1*norm2)
        norm1 = np.sqrt( self.integrate(n,n,weight="none") )
        norm2 = np.sqrt( self.integrate(m,m,weight="none") )
        realSpace = self.integrate(n,m,weight="support")/(norm1*norm2)

        # Contribution from Fourier domain
        #del X,Y,Z,func1,func2

        #N = self.mask.shape[0]
        #kx = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        #ky = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        #kz = np.linspace( -np.pi/2.0, np.pi/2.0, N )
        #KX,KY,KZ = np.meshgrid(kx,ky,kz)
        #func1 = self.basis.evalFourier( KX, KY, KZ, ix1, iy1, iz1 )
        #func2 = self.basis.evalFourier( KX, KY, KZ, ix2, iy2, iz2 )
        #norm1 = np.sqrt( np.sum( np.abs(func1)**2 ) )
        #norm2 = np.sqrt( np.sum( np.abs(func2)**2 ) )

        #insideMask = np.count_nonzero(self.mask)
        #fourierSpace = np.sum( np.conj(func1[self.mask==1])*func2[self.mask==1] )/(norm1*norm2)
        norm1 = np.sqrt( self.integrateFourier(n,n,weight="none") )
        norm2 = np.sqrt( self.integrateFourier(m,m,weight="none") )
        fourierSpace = self.integrateFourier(n,m,weight="mask")/(norm1*norm2)
        return 0.5*( realSpace + fourierSpace )

    def buildMatrix( self ):
        print ("Building matrix...")
        mat = np.zeros( (self.Nbasis**3,self.Nbasis**3) ) + 1j*np.zeros( (self.Nbasis**3,self.Nbasis**3) )
        for i in range(self.Nbasis**3):
            print ("Computing row %d of %d"%(i,self.Nbasis**3))
            for j in range(i,self.Nbasis**3):
                res = cnstpow.matrixElement( self, i, j )
                mat[i,j] = res[0]+1j*res[1]
                #mat[i,j] = self.operatorElement(i,j)
                mat[j,i] = np.conj(mat[i,j])
        print (np.sqrt( np.trace(np.conj(mat).T.dot(mat)) ))
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
