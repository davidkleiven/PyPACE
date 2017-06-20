#include "categorize.h"
#include <stdio.h>
#include <math.h>
//#define DEBUG

typedef unsigned char uint8;

uint8 getClusterID( double means[], int size, double newvalue )
{
  double min = 1E30;
  uint8 id = 0;
  for ( uint8 i=0;i<size;i++ )
  {
    if ( abs(means[i]-newvalue) < min )
    {
      id = i;
      min = abs(means[i]-newvalue);
    }
  }
  return id;
}

static PyObject* categorize( PyObject *self, PyObject *obj )
{
  #ifdef DEBUG
    printf("Entering categorize function\n");
  #endif

  PyObject *segmentor = NULL;
  // Parse the argument
  if ( !PyArg_ParseTuple( obj, "O", &segmentor) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the argmument" );
    return NULL;
  }

  #ifdef DEBUG
    printf("Input argument parsed\n");
  #endif
  assert( segmentor != NULL );

  // Check that the object satisfy the needs
  if ( !PyObject_HasAttrString(segmentor, "data") )
  {
    #ifdef DEBUG
      printf("Cannot find attribute data\n");
    #endif
    PyErr_SetString( PyExc_TypeError, "Object has no attribute data" );
    return NULL;
  }

  if ( !PyObject_HasAttrString(segmentor, "clusters") )
  {
    #ifdef DEBUG
      printf("Cannot find attribute clusters\n");
    #endif
    PyErr_SetString( PyExc_TypeError, "Object ha no attribute clusters" );
    return NULL;
  }

  if ( !PyObject_HasAttrString(segmentor,"means") )
  {
    #ifdef DEBUG
      printf("Cannot find attribute means\n");
    #endif
    PyErr_SetString( PyExc_TypeError, "Object has no attribute means" );
    return NULL;
  }
  #ifdef DEBUG
    printf("Start extracting numpy arrays\n");
  #endif

  // Extract the numpy arrays
  PyObject* npvalues = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"data"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  npy_intp* dims = PyArray_DIMS(npvalues);
  int nd = PyArray_NDIM(npvalues);

  PyObject* npcategories = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"clusters"), NPY_UINT8, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* dimsCat = PyArray_DIMS(npcategories);
  int ndcat = PyArray_NDIM( npcategories );

  PyObject* npmeans = PyArray_FROM_OTF( PyObject_GetAttrString(segmentor,"means"), NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY );
  npy_intp* nCat = PyArray_DIMS(npmeans);
  double newmeans[nCat[0]];
  int newmeansCount[nCat[0]];
  for ( int i=0;i<nCat[0];i++ )
  {
    newmeans[i] = 0.0;
    newmeansCount[i] = 0;
  }

  // Check consitency
  if ( nd != ndcat )
  {
    PyErr_SetString( PyExc_ValueError, "Categories and the data array has different number of dimensions");
    return NULL;
  }

  int totLengthData = 1;
  int totLengthCat = 1;
  for ( int i=0;i<nd;i++ )
  {
    totLengthData *= dims[i];
    totLengthCat *= dimsCat[i];
  }

  if ( totLengthData != totLengthCat )
  {
    PyErr_SetString( PyExc_ValueError, "Categories and the data array has different sizes");
    return NULL;
  }

  double* data = PyArray_DATA( npvalues );
  uint8* categories = PyArray_DATA( npcategories );
  int CONVERGED = 1;
  int NOT_CONVERGED = 0;
  int convergenceFlag = CONVERGED;

  // Update the means of the clusters
  double* mean = PyArray_DATA(npmeans);

  #ifdef DEBUG
    printf("Start putting voxels in categories\n");
  #endif
  for ( int i=0;i<totLengthData; i++ )
  {
    uint8 id = getClusterID( mean, nCat[0], data[i] );
    if ( id != categories[i] )
    {
      convergenceFlag = NOT_CONVERGED;
    }

    categories[i] = id;
    assert( id < nCat[0] );
    newmeans[id] += data[i];
    newmeansCount[id] += 1;
  }

  for ( int i=0;i<nCat[0];i++ )
  {
    mean[i] = newmeans[i]/newmeansCount[i];
  }

  #ifdef DEBUG
    printf("Returning from function categrozie\n");
  #endif
  if ( convergenceFlag == CONVERGED )
  {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* radialMean( PyObject* self, PyObject *args )
{
  PyObject *data = NULL;
  int Nbins = 0;

  if ( !PyArg_ParseTuple( args, "Oi", &data, &Nbins ) )
  {
    PyErr_SetString( PyExc_TypeError, "Wrong argument types in fucntion radialMean" );
    return NULL;
  }

  PyObject *npdata = PyArray_FROM_OTF( data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  npy_intp* dims = PyArray_DIMS( npdata );
  int nd = PyArray_NDIM( npdata );
  if ( nd != 3 )
  {
    PyErr_SetString( PyExc_ValueError, "Number of dimensions of python array has to be 3" );
    return NULL;
  }

  double rmax = 0.5*sqrt( dims[0]*dims[0] + dims[1]*dims[1] + dims[2]*dims[2] );
  double x0 = 0.5*dims[0];
  double y0 = 0.5*dims[1];
  double z0 = 0.5*dims[2];
  double rAvg[Nbins];
  int rCount[Nbins];

  // Initialize arrays with zeros
  for ( int i=0;i<Nbins;i++ )
  {
    rAvg[i] = 0.0;
    rCount[i] = 0;
  }

  // Perform radial averaging
  for ( int x=0;x<dims[0];x++ )
  for ( int y=0;y<dims[1];y++ )
  for ( int z=0;z<dims[2];z++ )
  {
    double r = sqrt( pow(x-x0,2) + pow(y-y0,2) + pow(z-z0,2) );
    int bin = (r*(Nbins-1)/rmax);
    if ( bin < Nbins )
    {
      rAvg[bin] += *((double *) PyArray_GETPTR3(npdata,x,y,z) );
      rCount[bin] += 1;
    }
  }

  // Create a numpy array to store the result in
  npy_intp length = Nbins;
  PyObject* result = PyArray_ZEROS( 1, &length, NPY_DOUBLE, 0 );
  double *rawptr = PyArray_DATA(result);
  Py_INCREF(result);
  // Divide by count
  for ( int i=0;i<Nbins;i++ )
  {
    rawptr[i] = rAvg[i]/rCount[i];
  }

  return result;
}

static PyMethodDef categorizeMethods[] = {
  {"categorize", categorize, METH_VARARGS, "Cluster data based on the closest mean. Arguments: Segmentor object"},
  {"radialMean", radialMean, METH_VARARGS, "Perform radial averaging on a 3D array. Arguments: 3D numpy array with data, number of bins"},
  {NULL,NULL,0,NULL}
};

static struct PyModuleDef categorizeModule = {
  PyModuleDef_HEAD_INIT,
  "categorize",
  NULL, // TODO: Write documentation string here
  -1,
  categorizeMethods
};

PyMODINIT_FUNC PyInit_categorize(void)
{
  PyObject* module = PyModule_Create( &categorizeModule );
  import_array();
  return module;
}
