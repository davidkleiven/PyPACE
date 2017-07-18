#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <assert.h>
#include <omp.h>

static PyObject* applyFourier( PyObject *self, PyObject *args )
{
  PyObject *missingData = NULL;
  PyObject *npft = NULL;

  if ( !PyArg_ParseTuple( args, "OO", &missingData, &npft) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the argmument" );
    return NULL;
  }

  PyObject* mask = PyArray_FROM_OTF( PyObject_GetAttrString(missingData,"mask"), NPY_UINT8, NPY_ARRAY_IN_ARRAY );
  PyObject *ft = PyArray_FROM_OTF( npft, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* dims = PyArray_DIMS(ft);
  npy_intp* dimsMask = PyArray_DIMS(mask);

  assert( dims[0] == dimsMask[0] );
  assert( dims[1] == dimsMask[1] );
  assert( dims[2] == dimsMask[2] );

  #pragma omp parallel for
  for ( int i=0;i<dims[0];i++ )
  {
    for ( int j=0;j<dims[1];j++)
    {
      for ( int k=0;k<dims[2];k++ )
      {

        uint8_t* val1 = (uint8_t *) PyArray_GETPTR3( mask, i, j, k);

        if ( *val1 == 1 )
        {
          // Project the result onto the subspace where the mask is zero
          double *value;
          value = (double *) PyArray_GETPTR3( ft, i,j,k );
          value[0] = 0.0;
          value[1] = 0.0;
        }
      }
    }
  }
  Py_DECREF(mask);
  return ft;
}

static PyObject* applyRealSpace( PyObject *self, PyObject *args )
{
  PyObject *missingData = NULL;
  PyObject *img = NULL;
  if ( !PyArg_ParseTuple( args, "OO", &missingData, &img) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the argmument" );
    return NULL;
  }

  PyObject *imgnp = PyArray_FROM_OTF( img, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  PyObject* support = PyArray_FROM_OTF( PyObject_GetAttrString(missingData,"support"), NPY_UINT8, NPY_ARRAY_IN_ARRAY );
  npy_intp* dims = PyArray_DIMS(imgnp);
  npy_intp* dimsSup = PyArray_DIMS(support);
  assert( dims[0] == dimsSup[0] );
  assert( dims[1] == dimsSup[1] );
  assert( dims[2] == dimsSup[2] );

  #pragma omp parallel for
  for ( int i=0;i<dims[0];i++ )
  for ( int j=0;j<dims[1];j++ )
  for ( int k=0;k<dims[2];k++ )
  {
    uint8_t* sup = (uint8_t *) PyArray_GETPTR3( support, i, j, k );
    double *imgval = (double *) PyArray_GETPTR3( imgnp, i, j, k );
    if (( *sup == 0 ) || ( *imgval < 0.0 ))
    {
      double *val = (double *) PyArray_GETPTR3( imgnp, i, j, k );
      *val = 0.0;
    }
  }
  Py_DECREF(support);
  return imgnp;
}

static PyObject* copyToRealPart( PyObject *self, PyObject *args )
{
  PyObject *source = NULL;
  PyObject *dest = NULL;
  if ( !PyArg_ParseTuple( args, "OO", &source, &dest) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the argmument" );
    return NULL;
  }

  PyObject* npsource = PyArray_FROM_OTF( source, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  PyObject* npdest = PyArray_FROM_OTF( dest, NPY_COMPLEX128, NPY_ARRAY_OUT_ARRAY );
  npy_intp* dims = PyArray_DIMS(npdest);

  #pragma omp parallel for
  for ( int i=0;i<dims[0]*dims[1]*dims[2];i++ )
  {
    int iz = i%dims[2];
    int iy = ( i/dims[2] )%dims[1];
    int ix = i/(dims[2]*dims[1]);
    double *sval = (double *) PyArray_GETPTR3( npsource, ix, iy, iz );
    double *dval = (double *) PyArray_GETPTR3( npdest, ix, iy, iz );
    dval[0] = *sval;
    dval[1] = 0.0;
  }
  //Py_DECREF(npdest);
  Py_DECREF(npsource);
  return npdest;
}

static PyMethodDef missingDataMethods[] = {
  {"applyFourier",applyFourier,METH_VARARGS,"Project the solution onto the space where the mask is zero"},
  {"applyRealSpace",applyRealSpace,METH_VARARGS,"Project the solution onto the space where the support is 1"},
  {"copyToRealPart", copyToRealPart, METH_VARARGS, "Copy array to the real part of an array"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef missingDataModule = {
    PyModuleDef_HEAD_INIT,
    "missingDataC",
    NULL, // TODO: Write documentation string here
    -1,
    missingDataMethods
  };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_missingdatac(void)
  {
    PyObject* module = PyModule_Create( &missingDataModule );
    import_array();
    return module;
  }
#else
  PyMODINIT_FUNC initmissingdatac(void)
  {
    Py_InitModule3( "missingdatac", missingDataMethods, "This the Python 2 version" );
    import_array();
  }
#endif
