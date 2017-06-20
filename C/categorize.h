#include <Python.h>
#include <numpy/ndarrayobject.h>

static PyObject* categorize( PyObject *self, PyObject *obj );
static PyObject* radialMean( PyObject *self, PyObject *args );
