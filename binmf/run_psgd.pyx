import numpy as np
cimport numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


cdef extern from "../src/psgd_sampled.c":
	void psgd(double *A, double *B, size_t dimA, size_t dimB, size_t k, size_t nnz,
			size_t *Xr_indptr, size_t *Xr_ind, double *Xr,
			double reg_param, size_t niter, int projected, int nthreads)

def fit_spfact(Xcsr, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B,
	double reg_param=1e-1, size_t niter=100, int nthreads=1, int projected=1):
	
	cdef size_t dimA = A.shape[0]
	cdef size_t dimB = B.shape[0]
	cdef size_t k = A.shape[1]
	cdef size_t nnz = Xcsr.nnz

	if Xcsr.data.dtype != np.float64:
		Xcsr.data = Xcsr.data.astype(ctypes.c_double)
	Xcsr.indptr = Xcsr.indptr.astype(ctypes.c_size_t)
	Xcsr.indices = Xcsr.indices.astype(ctypes.c_size_t)

	cdef np.ndarray[double, ndim=1] X_data = Xcsr.data
	cdef np.ndarray[size_t, ndim=1] X_indptr = Xcsr.indptr
	cdef np.ndarray[size_t, ndim=1] X_ind = Xcsr.indices

	psgd(&A[0,0], &B[0,0], dimA, dimB, k, nnz,
		&X_indptr[0], &X_ind[0], &X_data[0],
		reg_param, niter, projected, nthreads)


