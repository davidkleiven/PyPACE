cimport cython as ct
from cython.parallel cimport prange
cimport numpy as np
from libc cimport math as cmath
import multiprocessing as mp

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
