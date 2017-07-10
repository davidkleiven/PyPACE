import sys
sys.path.append("./")
import constrainedPowerBasis as cpb
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as interp
import constrainedpowerc as cnstpow
from scipy import ndimage as ndimg
from scipy import sparse as sp
import multiprocessing as mp
import itertools as itertls

class ConstrainedPower( object ):
    def __init__( self, mask, support, Nbasis=3 ):
        if ( Nbasis > 49 ):
            raise ValueError("According to the documentation on Hermite polynomials the integration routine has only been tested until 100."+
            "Here we need an integration scheme of 2*Nbasis+1 and this number should not exceed 100")

        self.mask = mask
        self.support = support
        self.recenter()
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
        self.integrationOrder = 2*self.Nbasis+1
        self.points, self.weights = np.polynomial.hermite.hermgauss( self.integrationOrder )

    def recenter( self ):
        com = np.array( ndimg.measurements.center_of_mass( self.support ) )
        center = int(self.support.shape[0]/2)
        self.support = ndimg.interpolation.shift( self.support, center-com )

        com = np.array( ndimg.measurements.center_of_mass( self.mask ) )
        center = int( self.mask.shape[0]/2 )
        self.mask = ndimg.interpolation.shift( self.mask, center-com )

    def computeScales( self ):
        N = self.support.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        z = np.linspace( -N/2, N/2, N )
        X,Y,Z = np.meshgrid(x,y,z)

        sumD = self.support.sum()
        scaleX = np.abs(X*self.support).max()
        scaleY = np.abs(Y*self.support).max()
        scaleZ = np.abs(Z*self.support).max()
        scaleX = np.sqrt( np.sum( X**2*self.support )/sumD )
        scaleY = np.sqrt( np.sum( Y**2*self.support )/sumD )
        scaleZ = np.sqrt( np.sum( Z**2*self.support )/sumD )
        del X,Y,Z
        N = self.mask.shape[0]
        kx = np.linspace( -np.pi, np.pi, N )

        # Focus on the central region
        width = N/4
        start = int( N/2-width/2 )
        end = int( N/2+width/2 )
        #kx = kx[start:end]
        #self.mask = self.mask[start:end,start:end,start:end]
        #self.maskInterp = interp.RegularGridInterpolator( (kx,kx,kx), self.mask, fill_value=0, bounds_error=False )
        KX,KY,KZ = np.meshgrid(kx,kx,kx)

        #scaleKx = np.abs( KX*(1-self.mask[start:end,start:end,start:end]) ).max()
        #scaleKy = np.abs( KY*(1-self.mask[start:end,start:end,start:end]) ).max()
        #scaleKz = np.abs( KZ*(1-self.mask[start:end,start:end,start:end]) ).max()
        sumM = np.sum(1-self.mask)
        scaleKx = np.sqrt( np.sum(KX**2 *(1-self.mask))/sumM )
        scaleKy = np.sqrt( np.sum(KY**2 *(1-self.mask))/sumM)
        scaleKz = np.sqrt( np.sum(KZ**2 *(1-self.mask))/sumM)
        return np.sqrt(scaleX/scaleKx), np.sqrt(scaleY/scaleKy), np.sqrt(scaleZ/scaleKz)

    def flattened2xyz( self, flattened ):
        iz = flattened%self.Nbasis
        iy = ( int( flattened/self.Nbasis )%self.Nbasis )
        ix = int( flattened/self.Nbasis**2 )
        return ix,iy,iz

    def xyz2Flattened( self, x, y, z ):
        return x*self.Nbasis**2 + y*self.Nbasis + z

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
        workers = mp.Pool( mp.cpu_count() )
        parallelMat = [MatrixOperator(i,self,self.Nbasis) for i in range(self.Nbasis**3)]
        rows = workers.map( innerLoopDense, parallelMat )
        for i in range(self.Nbasis**3):
            mat[i,i:] = rows[i]
            mat[i:,i] = np.conj(rows[i])
        return mat

    def createSparseList( self, bandwidth ):
        basisList = []
        for iz in range(-bandwidth,bandwidth):
            for iy in range(-bandwidth,bandwidth):
                if ( iz == 0 and iy == 0 ):
                    for ix in range(0,bandwidth):
                        basisList.append([ix,iy,iz])
                else:
                    for ix in range(-bandwidth,bandwidth):
                        basisList.append([ix,iy,iz])
        return basisList

    def convertBasisListToFlattenedIndex( self, i, basisList ):
        columns = []
        ix,iy,iz = self.flattened2xyz(i)
        for entry in basisList:
            indx = self.xyz2Flattened( ix+entry[0], iy+entry[1], iz+entry[2] )
            if ( indx < self.Nbasis**3 and indx >= i and not indx in columns ):
                columns.append( indx )
        return columns

    def checkIfAlreadyExists( self, newrow, newcol, allrows, allcols ):
        for i in range(len(allrows)):
            if ( newrow == allrows[i] and newcol == allcols[i] ):
                return True
        return False

    def buildMatrixSparse( self, bandwidth ):
        print ("Building sparse matrix with bandwidth=%d"%(bandwidth))
        if ( bandwidth > self.Nbasis ):
            raise ValueError("The bandwidth has to be smaller or equal to the number of basis functions in each direction")

        data = []
        row = []
        col = []
        N = self.Nbasis**3
        spList = self.createSparseList( bandwidth )
        """
        parMat = [ MatrixOperatorSparse(i, self, spList) for i in range(N) ]
        workers = mp.Pool( mp.cpu_count() )
        result = workers.map( innerLoopSparse, parMat )
        for pMat in result:
            data += pMat.data
            row += pMat.rows
            col += pMat.cols
        """
        for i in range(N):
            print ("Computing row %d of %d"%(i,self.Nbasis**3))
            cols = self.convertBasisListToFlattenedIndex( i, spList )
            for j in cols:
                res = cnstpow.matrixElement( self, i, j )
                data.append( res[0]+1j*res[1] )
                row.append(i)
                col.append(j)
                if ( i != j ):
                    data.append( res[0]-1j*res[1] )
                    row.append(j)
                    col.append(i)
        # Creating a intermediate DOK matrix removes duplicates and using the last one
        # https://stackoverflow.com/questions/28677162/ignoring-duplicate-entries-in-sparse-matrix
        return sp.csc_matrix( (data,(row,col)) )

    def plotEigenvalues( self ):
        if ( self.eigval is None ):
            return None
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.eigval)
        return fig

    def solve( self, mode="dense", bandwidth=1, plotMatrix=False, fracEigmodes=0.5 ):
        if ( mode == "dense" ):
            mat = self.buildMatrix()
            plt.matshow(np.abs(mat))
            plt.colorbar()
            plt.show()
            print ("Solving eigensystem...")
            self.eigval, self.eigvec = np.linalg.eigh(mat)
        elif ( mode == "sparse" ):
            mat = self.buildMatrixSparse( bandwidth )
            if ( plotMatrix ):
                plt.matshow( np.abs(mat.todense()) )
                plt.colorbar()
                plt.show()
            print ("Solving eigensystem...")
            self.eigval, self.eigvec = sp.linalg.eigsh( mat, k=int(fracEigmodes*self.Nbasis**3), which="SM" )
        else:
            raise ValueError("Mode has to be either dense or sparse")
        return self.eigval, self.eigvec

    def getEigenModeReal( self, modeNum, x, y, z ):
        mode = self.eigvec[0,modeNum]*self.basis.eval(x,y,z,0,0,0)
        for i in range(1,self.eigvec.shape[0]):
            nx,ny,nz = self.flattened2xyz( i )
            mode += self.eigvec[i,modeNum]*self.basis.eval(x,y,z,nx,ny,nz)
        return mode

    def getEigenModeFourier( self, modeNum, kx, ky, kz ):
        mode = self.eigvec[0,modeNum]*self.basis.evalFourierWithWeight(kx,ky,kz,0,0,0)
        for i in range(1,self.eigvec.shape[0] ):
            nx,ny,nz = self.flattened2xyz( i )
            mode += self.eigvec[i,modeNum]*self.basis.evalFourierWithWeight(kx,ky,kz,nx,ny,nz)
        return mode


class MatrixOperator( object ):
    def __init__( self, row, cnstPower, Nbasis ):
        self.row = row
        self.rowval = np.zeros( Nbasis**3 - row, dtype=np.complex64 )
        self.cnstPower = cnstPower
        self.Nbasis = Nbasis

class MatrixOperatorSparse( object ):
    def __init__( self, row, cnstPower, spList ):
        self.row = row
        self.cnstPower = cnstPower
        self.spList = spList
        self.data = []
        self.rows = []
        self.cols = []

def innerLoopDense( parMat ):
    i = parMat.row
    #mat = args[1]
    for j in range(i,parMat.Nbasis**3):
        res = cnstpow.matrixElement( parMat.cnstPower, i, j )
        parMat.rowval[j-i] = res[0]+1j*res[1]
    return parMat.rowval

def innerLoopSparse( pMatSparse ):
    if ( not isinstance(pMatSparse,MatrixOperatorSparse) ):
        raise TypeError("Argument has to be of type MatrixOperatorSparse")

    i = pMatSparse.row
    jval = pMatSparse.cnstPower.convertBasisListToFlattenedIndex( pMatSparse.row, pMatSparse.spList )
    for j in jval:
        res = cnstpow.matrixElement( pMatSparse.cnstPower, i, j )
        pMatSparse.data.append( res[0]+1j*res[1] )
        pMatSparse.rows.append(i)
        pMatSparse.cols.append(j)
        if ( i != j ):
            pMatSparse.data.append( res[0]-1j*res[1] )
            pMatSparse.rows.append(j)
            pMatSparse.cols.append(i)
    return pMatSparse
