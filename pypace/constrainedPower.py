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
    def __init__( self, mask, support, voxelsize, Nbasis=3 ):
        if ( Nbasis > 49 ):
            raise ValueError("According to the documentation on Hermite polynomials the integration routine has only been tested until 100."+
            "Here we need an integration scheme of 2*Nbasis+1 and this number should not exceed 100")

        self.voxelsize = voxelsize
        self.mask = mask
        self.support = support
        self.recenter()
        self.scaleX, self.scaleY, self.scaleZ = self.computeScales()
        self.basis = cpb.Basis( self.scaleX, self.scaleY, self.scaleZ )
        self.Nbasis = Nbasis
        self.eigval = None
        self.eigvec = None
        self.integrationOrder = 2*self.Nbasis+1
        self.integrationOrder = 61
        self.points, self.weights = np.polynomial.hermite.hermgauss( self.integrationOrder )

    def recenter( self ):
        print ("Centering the data...")
        com = np.array( ndimg.measurements.center_of_mass( self.support ) )
        center = int(self.support.shape[0]/2)
        #self.support = ndimg.interpolation.shift( self.support, center-com )
        for i in range(3):
            self.support = np.roll( self.support, int(center-com[i]), axis=i )

        com = np.array( ndimg.measurements.center_of_mass( self.mask ) )
        center = int( self.mask.shape[0]/2 )
        #self.mask = ndimg.interpolation.shift( self.mask, center-com )
        for i in range(3):
            self.mask = np.roll( self.mask, int(center-com[i]), axis=i)

    def computeScales( self ):
        print ("Computing scales...")
        # Approximate these value by using meshgrid on a downsamples version of the arrays
        reduction = 16
        N = self.support.shape[0]
        x = np.linspace( -N*self.voxelsize/2, N*self.voxelsize/2, N/reduction )
        y = np.linspace( -N*self.voxelsize/2, N*self.voxelsize/2, N/reduction )
        z = np.linspace( -N*self.voxelsize/2, N*self.voxelsize/2, N/reduction )
        X,Y,Z = np.meshgrid(x,y,z)

        sup = self.support[::reduction,::reduction,::reduction]
        sumD = sup.sum()
        scaleX = np.sqrt( np.sum( X**2*sup )/sumD )
        scaleY = np.sqrt( np.sum( Y**2*sup )/sumD )
        scaleZ = np.sqrt( np.sum( Z**2*sup )/sumD )
        del X,Y,Z
        N = self.mask.shape[0]
        kx = np.linspace( -np.pi/self.voxelsize, np.pi/self.voxelsize, N/reduction )

        # Focus on the central region
        width = N/4
        start = int( N/2-width/2 )
        end = int( N/2+width/2 )
        KX,KY,KZ = np.meshgrid(kx,kx,kx)

        msk = self.mask[::reduction,::reduction,::reduction]
        sumM = np.sum(1-msk)
        scaleKx = np.sqrt( np.sum(KX**2 *(1-msk))/sumM )
        scaleKy = np.sqrt( np.sum(KY**2 *(1-msk))/sumM)
        scaleKz = np.sqrt( np.sum(KZ**2 *(1-msk))/sumM)
        print ("Scales computed...")
        return np.sqrt(scaleX/scaleKx), np.sqrt(scaleY/scaleKy), np.sqrt(scaleZ/scaleKz)

    def flattened2xyz( self, flattened ):
        iz = flattened%self.Nbasis
        iy = ( int( flattened/self.Nbasis )%self.Nbasis )
        ix = int( flattened/self.Nbasis**2 )
        return ix,iy,iz

    def xyz2Flattened( self, x, y, z ):
        return x*self.Nbasis**2 + y*self.Nbasis + z

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
            #self.eigval, self.eigvec = np.linalg.eigh(mat)
            self.eigvec, self.eigval, v = np.linalg.svd(mat)
            self.eigval = self.eigval[::-1]
            self.eigvec = np.fliplr(self.eigvec)
            #self.eigval[self.eigval>0.0] = 1.0/self.eigval[self.eigval>0.0]
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
        print (self.eigvec[:,0])
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
