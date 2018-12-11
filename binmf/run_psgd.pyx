import numpy as np
cimport numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


IF UNAME_SYSNAME == "Windows":
	obj_ind_type = ctypes.c_long
	ctypedef long ind_type
ELSE:
	obj_ind_type = ctypes.c_size_t
	ctypedef size_t ind_type

cdef extern from "psgd_sampled.c":
	void psgd(double *A, double *B, ind_type dimA, ind_type dimB, ind_type k, ind_type nnz,
			ind_type *Xr_indptr, ind_type *Xr_ind, double *Xr,
			double reg_param, ind_type niter, int projected, int nthreads,
			double *buffer_B, double *Anew, double *Bnew, ind_type *Acnt, ind_type *Bcnt)

def fit_spfact(Xcsr, np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=2] B,
	double reg_param=1e-1, ind_type niter=100, int nthreads=1, int projected=1):
	
	cdef ind_type dimA = A.shape[0]
	cdef ind_type dimB = B.shape[0]
	cdef ind_type k = A.shape[1]
	cdef ind_type nnz = Xcsr.nnz

	if Xcsr.data.dtype != np.float64:
		Xcsr.data = Xcsr.data.astype('float64')
	Xcsr.indptr = Xcsr.indptr.astype(obj_ind_type)
	Xcsr.indices = Xcsr.indices.astype(obj_ind_type)

	cdef np.ndarray[double, ndim=1] X_data = Xcsr.data
	cdef np.ndarray[ind_type, ndim=1] X_indptr = Xcsr.indptr
	cdef np.ndarray[ind_type, ndim=1] X_ind = Xcsr.indices

	cdef np.ndarray[double, ndim=2] Anew = np.empty((dimA, k), dtype='float64')
	cdef np.ndarray[double, ndim=2] Bnew = np.empty((dimB, k), dtype='float64')

	cdef np.ndarray[ind_type, ndim=1] Acnt = np.empty(dimA, dtype=obj_ind_type)
	cdef np.ndarray[ind_type, ndim=1] Bcnt = np.empty(dimB, dtype=obj_ind_type)

	cdef np.ndarray[double, ndim=1] buffer_B = np.empty(dimB * (k + 1) * nthreads, dtype='float64')

	psgd(&A[0,0], &B[0,0], dimA, dimB, k, nnz,
		&X_indptr[0], &X_ind[0], &X_data[0],
		reg_param, niter, projected, nthreads,
		&buffer_B[0], &Anew[0,0], &Bnew[0,0], &Acnt[0], &Bcnt[0])


