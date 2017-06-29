cimport cython as ct
from cython.parallel cimport prange
cimport numpy as np
from cython cimport parallel
from libc cimport math as cmath
import multiprocessing as mp
import numpy as regNP # Regular numpy

cdef int nproc = mp.cpu_count()

cdef double cabs( np.complex128_t z ) nogil:
    return cmath.sqrt( z.real**2 + z.imag**2 )

@ct.boundscheck(False)
@ct.wraparound(False)
def applyFourierConstraint( data, measured ):
    cdef np.ndarray[np.complex128_t] dataR = data.ravel()
    cdef np.ndarray[np.float64_t] measuredR = measured.ravel()
    cdef int i
    cdef int size = data.size

    cdef double absVal
    cdef double meas
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc):
        absVal = cabs(dataR[i])
        meas = measuredR[i]
        if ( absVal > 0.0 ):
            dataR[i] = dataR[i]*meas/absVal
        else:
            dataR[i] = 0.0
    return data

@ct.boundscheck(False)
@ct.wraparound(False)
def applyHybridConstraint( data, mask, beta, lastObject ):
    cdef np.ndarray[np.complex128_t] dataR = data.ravel()
    cdef np.ndarray[np.float64_t] dataRealR = data.real.ravel()
    cdef np.ndarray[np.uint8_t] maskR = mask.ravel()
    cdef double betaC = beta
    cdef np.ndarray[np.complex128_t] lastObjectR = lastObject.ravel()

    cdef int i
    cdef int size = data.size

    for i in prange(size, nogil=True, schedule="static", num_threads=nproc):
        if ( maskR[i] == 0 or dataRealR[i] < 0.0 ):
            dataR[i] = dataR[i] - betaC*lastObjectR[i]
    return data

@ct.boundscheck(False)
@ct.wraparound(False)
def getThresholdMask( data, mask, threshold ):
    cdef np.ndarray[np.float64_t] dataR = data.ravel()
    cdef np.ndarray[np.uint8_t] maskR = mask.ravel()
    cdef double thresC = threshold
    cdef int i
    cdef int size = data.size

    for i in prange(size, nogil=True, schedule="static", num_threads=nproc):
        if ( dataR[i] > thresC ):
            maskR[i] = 1
        else:
            maskR[i] = 0
    return mask


@ct.boundscheck(False)
@ct.wraparound(False)
def copy( fromData, toData ):
    cdef np.ndarray[np.complex128_t] fromR = fromData.ravel()
    cdef np.ndarray[np.complex128_t] toR = toData.ravel()

    cdef int i
    cdef int size = fromData.size
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc):
        toR[i] = fromR[i]
    return toData

@ct.boundscheck(False)
@ct.wraparound(False)
def max( array ):
    cdef np.ndarray[np.float64_t] arrayR = array.ravel()
    cdef np.ndarray[np.float64_t] maxVals = regNP.zeros(nproc)-1E30
    cdef int i
    cdef int size = array.size
    cdef double maxval = -1E30
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc ):
        if ( arrayR[i] > maxVals[parallel.threadid()] ):
            maxVals[parallel.threadid()] = arrayR[i]
    return regNP.max(maxVals)

@ct.boundscheck(False)
@ct.wraparound(False)
def meanSquareError( data1, data2 ):
    assert( data1.size == data2.size )
    cdef np.ndarray[np.float64_t] data1R = data1.ravel()
    cdef np.ndarray[np.float64_t] data2R = data2.ravel()
    cdef int i
    cdef int size = data1.size
    cdef np.ndarray[np.float64_t] totalSum = regNP.zeros(nproc)
    cdef np.ndarray[np.float64_t] diff = regNP.zeros(nproc)
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc ):
        diff[parallel.threadid()] = data1R[i] - data2R[i]
        totalSum[parallel.threadid()] = totalSum[parallel.threadid()] + diff[parallel.threadid()]*diff[parallel.threadid()]
    return regNP.sqrt( regNP.sum(totalSum) )/size

@ct.boundscheck(False)
@ct.wraparound(False)
def modulus( dataIn, dataOut ):
    assert( dataIn.size == dataOut.size )
    cdef np.ndarray[np.complex128_t] dataInR = dataIn.ravel()
    cdef np.ndarray[np.float64_t] dataOutR = dataOut.ravel()
    cdef int i
    cdef int size = dataIn.size
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc ):
        dataOutR[i] = cabs(dataInR[i])
    return dataOut

@ct.boundscheck(False)
@ct.wraparound(False)
def mean( data ):
    cdef np.ndarray[np.float64_t] dataR = data.ravel()
    cdef int i
    cdef int size = data.size
    cdef np.ndarray[np.float64_t] totSum = regNP.zeros(nproc)
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc ):
        totSum[parallel.threadid()] = totSum[parallel.threadid()] + dataR[i]
    return regNP.sum(totSum)/size

@ct.boundscheck(False)
@ct.wraparound(False)
def meanWithMask( data, mask ):
    assert( data.size == mask.size )
    cdef np.ndarray[np.float64_t] dataR = data.ravel()
    cdef np.ndarray[np.uint8_t] maskR = mask.ravel()
    cdef int i
    cdef int size = data.size
    cdef np.ndarray[np.float64_t] totSum = regNP.zeros(nproc)
    cdef np.ndarray[np.float64_t] counter = regNP.zeros(nproc)
    for i in prange(size, nogil=True, schedule="static", num_threads=nproc ):
        if ( maskR[i] == 1 ):
            totSum[parallel.threadid()] = totSum[parallel.threadid()] + dataR[i]
            counter[parallel.threadid()] = counter[parallel.threadid()] + 1
    return regNP.sum(totSum)/regNP.sum(counter)
