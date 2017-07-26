cimport cython as ct
from cython.parallel cimport prange
cimport numpy as np
import numpy as np
from cython cimport parallel
from libc cimport math as cmath
import multiprocessing as mp

cdef int nproc = mp.cpu_count()

@ct.boundscheck(False)
@ct.wraparound(False)
def projectCluster( clusters, id, output, axis ):
    assert( len(clusters.shape) == 3 )
    assert( len(output.shape) == 2)
    assert( clusters.shape[0] == clusters.shape[1] )
    assert( clusters.shape[0] == clusters.shape[2] )
    assert( clusters.shape[0] == output.shape[0] )
    assert( clusters.shape[0] == output.shape[1] )
    cdef np.ndarray[np.uint8_t, ndim=3] clustersC = clusters
    cdef int idC = id
    cdef np.ndarray[np.uint8_t, ndim=2] outputC = output
    cdef int N = clusters.shape[0]
    cdef int ix, iy, iz
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                if ( clustersC[ix,iy,iz] == idC and axis==0 ):
                    outputC[iy,iz] = outputC[iy,iz]+1
                elif ( clustersC[ix,iy,iz] == idC and axis==1 ):
                    outputC[ix,iz] = outputC[ix,iz]+1
                elif ( clustersC[ix,iy,iz] == idC and axis==2 ):
                    outputC[ix,iy] = outputC[ix,iy]+1
    return outputC

@ct.boundscheck(False)
@ct.wraparound(False)
def averageAzimuthalClusterWeightsAroundX( clusters, clusterID, output, mask ):
    assert( len(output.shape) == 2 )
    assert( len(clusters.shape)  == 3 )
    assert( clusters.shape[0] == clusters.shape[1] )
    assert( clusters.shape[0] == clusters.shape[2] )
    assert( clusters.shape[0] == output.shape[0] )
    assert( clusters.shape[0] == output.shape[1] )

    cdef np.ndarray[np.float64_t, ndim=2] outputR = output
    cdef np.ndarray[np.uint8_t,ndim=3] clustersR = clusters
    cdef np.ndarray[np.uint8_t,ndim=3] maskC = mask

    cdef int N = clusters.shape[0]
    cdef int cID = clusterID
    cdef float qy, qz, qperp, weight
    cdef int qperpInt
    cdef center=N/2
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                if ( maskC[ix,iy,iz] == 0 ):
                    continue
                if ( clustersR[ix,iy,iz] == cID ):
                    qy = iy-N/2
                    qz = iz-N/2
                    qperp = cmath.sqrt(qy*qy+qz*qz)
                    qperpInt = <int>qperp # Rounds to integer below
                    if ( qperpInt < (N-1)/2 ):
                        weight = qperp-qperpInt
                        outputR[ix,center+qperpInt] = outputR[ix,center+qperpInt]+1.0-weight
                        outputR[ix,center+qperpInt+1] = outputR[ix,center+qperpInt+1] + weight
                    elif ( qperpInt < N/2 ):
                        outputR[ix,center+qperpInt] = outputR[ix,center+qperpInt] + 1.0
    return output

@ct.boundscheck(False)
@ct.wraparound(False)
def azimuthalAverageX( data3D, output, mask ):
    assert( len(data3D.shape) == 3 )
    assert( len(output.shape) == 2 )
    assert( data3D.shape[0] == data3D.shape[1] )
    assert( data3D.shape[0] == data3D.shape[2] )
    assert( data3D.shape[0] == output.shape[0] )
    assert( data3D.shape[0] == output.shape[1] )
    cdef np.ndarray[np.float64_t,ndim=3] data3DC = data3D
    cdef np.ndarray[np.float64_t,ndim=2] outputC = output
    cdef np.ndarray[np.uint8_t,ndim=3] maskC = mask
    cdef int N = data3D.shape[0]
    cdef float qy, qz, qperp, weight
    cdef int qperpInt
    cdef int center = N/2

    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                if ( maskC[ix,iz,iz] == 0 ):
                    continue
                qy = iy-N/2
                qz = iz-N/2
                qperp = cmath.sqrt( qy*qy + qz*qz )
                qperpInt = <int>qperp # Rounds to integer below
                if ( qperpInt < N-1 ):
                    weight = qperp-qperpInt
                    outputC[ix,center+qperpInt] = outputC[ix,center+qperpInt] + data3DC[ix,iy,iz]*(1.0-weight)
                    outputC[ix,center+qperpInt+1] = outputC[ix,center+qperpInt+1] + data3DC[ix,iy,iz]*weight
                elif ( qperpInt < N ):
                    outputC[ix,center+qperpInt] = outputC[ix,center+qperpInt] + data3DC[ix,iy,iz]
    return output

@ct.boundscheck(False)
@ct.wraparound(False)
def maskedSumOfSquares( reference, data, mask ):
    cdef np.ndarray[np.float64_t] ref = reference.ravel()
    cdef np.ndarray[np.float64_t] dataR = data.ravel()
    cdef np.ndarray[np.uint8_t] maskR = mask.ravel()

    cdef int N = data.size
    cdef int i
    cdef np.ndarray[np.float64_t] sumsq = np.zeros(nproc)
    for i in prange(N, nogil=True, num_threads=nproc):
        if ( maskR[i] == 1 ):
            sumsq[parallel.threadid()] = sumsq[parallel.threadid()] + cmath.pow( ref[i]-dataR[i], 2 )
    return np.sum(sumsq)
