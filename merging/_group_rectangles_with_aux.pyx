""" Example of wrapping a C function that takes C float arrays as input using
    the Numpy declarations from Cython """

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

# cdefine the signature of our c function
cdef extern from "group_rectangles_with_aux.h":
    int groupRectanglesWithAux(float* input, int group_threshold, float eps, \
        float* output, int* scores, int nrows, int ncols)

def execute(list linput, int group_threshold, float eps = 0.2):
    cdef int nrows = len(linput)
    cdef int ncols = np.PyArray_DIM(linput[0], 0)
    cdef np.ndarray[float, ndim=2, mode="c"] input = np.empty((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        input[i, :] = linput[i]
    cdef np.ndarray[float, ndim=2, mode="c"] output = np.empty((nrows, ncols), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] scores = np.empty((nrows, 1), dtype=np.int32)

    cdef int num_output = groupRectanglesWithAux(
        <float*> np.PyArray_DATA(input),
        group_threshold,
        eps,
        <float*> np.PyArray_DATA(output),
        <int*> np.PyArray_DATA(scores),
        nrows,
        ncols)

    if ncols > 4:
      aux = output[:num_output, 4:]
    else:
      aux = None
    return (np.rint(output[:num_output, :4]).astype(np.int32), aux, scores[:num_output])
