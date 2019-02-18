extern "C" {
	#include <stddef.h>
	#include <R_ext/BLAS.h>
	void psgd(double *A, double *B, size_t dimA, size_t dimB, size_t k, size_t nnz,
		size_t *X_indptr, size_t *X_ind, double *Xr,
		double reg_param, size_t niter, int projected, int nthreads);
}
#include <Rcpp.h>

// [[Rcpp::export]]
void r_wrapper_binmf(Rcpp::NumericVector A, Rcpp::NumericVector B, size_t dimA, size_t dimB, size_t k,
	Rcpp::NumericVector Xr, Rcpp::IntegerVector Xind, Rcpp::IntegerVector Xindptr, size_t nnz,
	double reg_param, size_t niter, int projected, int nthreads)
{
	/* Convert CSR matrix indices to size_t */
	std::vector<size_t> X_ind;
	std::vector<size_t> X_indptr;
	X_ind.reserve(dimA + 1);
	X_indptr.reserve(nnz);
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t i = 0; i < dimA + 1; i++) { X_ind[i] = Xind[i]; }
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (size_t i = 0; i < nnz; i++) { X_indptr[i] = Xindptr[i]; }

	/* Run procedure */
	psgd(A.begin(), B.begin(), dimA, dimB, k, nnz,
		(size_t*) &X_indptr[0], (size_t*) &X_ind[0], Xr.begin(),
		reg_param, niter, projected, nthreads);

	/* Note: C++ refuses to acknowledge that the vectors of type unsigned long are equivalent to size_t,
	   so don't use method .begin with the indices arrays */
}

// [[Rcpp::export]]
void predict_multiple(Rcpp::NumericVector A, Rcpp::NumericVector B, int k, size_t npred,
	Rcpp::IntegerVector ia, Rcpp::IntegerVector ib, Rcpp::NumericVector out, int nthreads)
{
	int one = 1;
	#pragma omp parallel for shared(npred, out, A, ia, B, ib, k) num_threads(nthreads)
	for (size_t i = 0; i < npred; i++) { out[i] = ddot_(&k, &A[ia[i] * k], &one, &B[ib[i] * k], &one); }
}
