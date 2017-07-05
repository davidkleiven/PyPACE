#include <Python.h>
#include "shellKmeans.hpp"
#include <numpy/ndarrayobject.h>

bool isSegmentorObject( PyObject *segmentor )
{
  bool ok = true;
  ok = ok && PyObject_HasAttrString(segmentor, "data");
  ok = ok && PyObject_HasAttrString(segmentor, "clusters");
  ok = ok && PyObject_HasAttrString(segmentor, "means");
  ok = ok && PyObject_HasAttrString(segmentor, "shellradii");
  return ok;
}

static PyObject* categorize( PyObject *self, PyObject *obj )
{
  PyObject *segmentor = NULL;
  PyObject *weights = NULL;
  // Parse the argument
  if ( !PyArg_ParseTuple( obj, "OO", &segmentor, &weights) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the argmument" );
    return NULL;
  }

  if ( not isSegmentorObject(segmentor) )
  {
    PyErr_SetString( PyExc_TypeError, "Argument segmentor does not satisfy the required attribues");
    return NULL;
  }

  ShellKmeans kmeans;
  PyObject* npvalues = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"data"), NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* dims = PyArray_DIMS(npvalues);
  kmeans.data = static_cast<double*>( PyArray_GETPTR3(npvalues,0,0,0) );
  kmeans.N = dims[0];

  PyObject* npclust = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"clusters"), NPY_UINT8, NPY_ARRAY_INOUT_ARRAY );
  kmeans.clusters = static_cast<unsigned char*>( PyArray_GETPTR3(npclust,0,0,0) );

  PyObject* npweights = PyArray_FROM_OTF( weights, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  kmeans.weights = static_cast<double*>( PyArray_GETPTR1(npweights,0) );

  PyObject* npmeans = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"means"), NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* npNclust = PyArray_DIMS(npmeans);
  kmeans.means = static_cast<double*>( PyArray_GETPTR1(npmeans,0) );

  PyObject* npshellradii = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"shellradii"), NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  kmeans.shellradii = static_cast<double*>( PyArray_GETPTR1(npshellradii,0) );

  kmeans.nClusters = npNclust[0];


  switch( kmeans.run() )
  {
    case ShellKmeans::converged_t::CONVERGED:
      return Py_True;
    case ShellKmeans::converged_t::NOTCONVERGED:
      return Py_False;
  }
  return Py_False;
}

static PyMethodDef categorizeMethods[] = {
  {"categorize", categorize, METH_VARARGS, "Cluster data based on the closest mean. Arguments: Segmentor object"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef categorizeModule = {
    PyModuleDef_HEAD_INIT,
    "shellCategorize",
    NULL, // TODO: Write documentation string here
    -1,
    categorizeMethods
  };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_shellCategorize(void)
  {
    PyObject* module = PyModule_Create( &categorizeModule );
    import_array();
    return module;
  }
#else
  PyMODINIT_FUNC initshellCategorize(void)
  {
    Py_InitModule3( "shellCategorize", categorizeMethods, "This the Python 2 version" );
    import_array();
  }
#endif
