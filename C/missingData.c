#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <assert.h>

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
  PyObject *ft = PyArray_FROM_OTF( npft, NPY_COMPLEX64, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* dims = PyArray_DIMS(ft);
  npy_intp* dimsMask = PyArray_DIMS(mask);

  for ( int i=0;i<dims[0];i++ )
  {
    int imask1 = dimsMask[0]/2 + i;
    int imask2 = dimsMask[0]/2-i;
    assert( imask1 < dimsMask[0] );
    assert( imask2 >= 0 );
    for ( int j=0;j<dims[1];j++)
    {
      int jmask1 = dimsMask[1]/2 + j;
      int jmask2 = dimsMask[1]/2 - j;
      assert( jmask1 < dimsMask[1] );
      assert( jmask2 >= 0 );
      for ( int k=0;k<dims[2];k++ )
      {
        int kmask1 = dimsMask[2]/2 + k;
        int kmask2 = dimsMask[2]/2 - k;
        assert( kmask1 < dimsMask[2] );
        assert( kmask2 >= 0 );
        uint8_t* val1 = (uint8_t *) PyArray_GETPTR3( mask, imask1, jmask1, kmask1 );
        uint8_t* val2 = (uint8_t *) PyArray_GETPTR3( mask, imask2, jmask2, kmask2 );
        if (( *val1 == 1 ) || ( *val2 == 1 ))
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
  Py_RETURN_TRUE;
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

  for ( int i=0;i<dims[0];i++ )
  for ( int j=0;j<dims[1];j++ )
  for ( int k=0;k<dims[2];k++ )
  {
    uint8_t* sup = (uint8_t *) PyArray_GETPTR3( support, i, j, k );
    if ( *sup == 0 )
    {
      double *val = (double *) PyArray_GETPTR3( imgnp, i, j, k );
      *val = 0.0;
    }
  }
  Py_RETURN_TRUE;
}

static PyMethodDef missingDataMethods[] = {
  {"applyFourier",applyFourier,METH_VARARGS,"Project the solution onto the space where the mask is zero"},
  {"applyRealSpace",applyRealSpace,METH_VARARGS,"Project the solution onto the space where the support is 1"},
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
