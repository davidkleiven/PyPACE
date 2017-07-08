#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "hermiteOperatorMatrix.hpp"
#include <iostream>
//#define DEBUG

using namespace std;

static PyObject* matrixElement( PyObject *self, PyObject *args )
{
  PyObject *cnstPower = nullptr;
  int n, m;

  if ( !PyArg_ParseTuple( args, "Oii", &cnstPower, &n, &m) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse arguments in matrixElement" );
    return NULL;
  }

  double scaleX = PyFloat_AsDouble( PyObject_GetAttrString(cnstPower,"scaleX") );
  double scaleY = PyFloat_AsDouble( PyObject_GetAttrString(cnstPower,"scaleY") );
  double scaleZ = PyFloat_AsDouble( PyObject_GetAttrString(cnstPower,"scaleZ") );
  int nbasis = PyInt_AsLong( PyObject_GetAttrString(cnstPower,"Nbasis") );

  #ifdef DEBUG
    cout << "ScaleX=" << scaleX << " scaleY=" << scaleY << " scaleZ=" << scaleZ << endl;
    cout << "Nbasis=" << nbasis << endl;
  #endif

  // Extract numpy-arrays
  PyObject *support = PyArray_FROM_OTF( PyObject_GetAttrString(cnstPower,"support"), NPY_UINT8, NPY_ARRAY_IN_ARRAY );
  uint8_t *supportPtr = static_cast<uint8_t*>( PyArray_GETPTR3( support, 0,0,0 ) );
  PyObject *mask = PyArray_FROM_OTF( PyObject_GetAttrString(cnstPower,"mask"), NPY_UINT8, NPY_ARRAY_IN_ARRAY );
  uint8_t *maskPtr = static_cast<uint8_t*>( PyArray_GETPTR3( mask, 0, 0, 0 ) );
  PyObject *points = PyArray_FROM_OTF( PyObject_GetAttrString(cnstPower,"points"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  double *pointsPtr = static_cast<double*>( PyArray_GETPTR1( points, 0 ) );
  PyObject *weights = PyArray_FROM_OTF( PyObject_GetAttrString(cnstPower,"weights"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  double *weightsPtr = static_cast<double*>( PyArray_GETPTR1( weights, 0 ) );

  // Extract dimensions
  npy_intp* dimsSupport= PyArray_DIMS( support );
  int Nsup = dimsSupport[0];
  npy_intp* dimsMask = PyArray_DIMS( mask );
  int Nmask = dimsMask[0];
  npy_intp* dimsWeights = PyArray_DIMS( weights );
  int integorder = dimsWeights[0];

  HermiteOperatorMatrix op( weightsPtr, pointsPtr, integorder, nbasis, supportPtr, maskPtr, Nsup, Nmask, scaleX, scaleY, scaleZ );

  double real, imag;
  op.matrixElement( n, m, real, imag );
  return Py_BuildValue( "[d,d]", real, imag );
}

static PyMethodDef constPowerMethods[] = {
  {"matrixElement",matrixElement, METH_VARARGS, "Computes one matrix element of the constrained power operator"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef constrainedPowerModule = {
    PyModuleDef_HEAD_INIT,
    "constrainedPowerC",
    NULL, // TODO: Write documentation string here
    -1,
    constPowerMethod
  };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_constrainedpowerc(void)
  {
    PyObject* module = PyModule_Create( &constrainedPowerModule );
    import_array();
    return module;
  }
#else
  PyMODINIT_FUNC initconstrainedpowerc(void)
  {
    Py_InitModule3( "constrainedpowerc", constPowerMethods, "This the Python 2 version" );
    import_array();
  }
#endif
