 /* Sparse Binary Matrix Factorization

 Fit through projected sub-gradient descent, sampling missing entries at random in each iteration.
 Writen for C99 standard.

 Copyright David Cortes 2018 */

#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

/* BLAS functions
https://stackoverflow.com/questions/52905458/link-cython-wrapped-c-functions-against-blas-from-numpy/52913120#52913120
 */
double ddot(int *N, double *DX, int *INCX, double *DY, int *INCY);
void daxpy(int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY);
void dscal(int *N, double *DA, double *DX, int *INCX);
double dnrm2(int *N, double *X, int *INCX);


#ifndef ddot
double ddot_(int *N, double *DX, int *INCX, double *DY, int *INCY);
#define ddot(N, DX, INCX, DY, INCY) ddot_(N, DX, INCX, DY, INCY)
#endif

#ifndef daxpy
void daxpy_(int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY);
#define daxpy(N, DA, DX, INCX, DY, INCY) daxpy_(N, DA, DX, INCX, DY, INCY)
#endif

#ifndef dscal
void dscal_(int *N, double *DA, double *DX, int *INCX);
#define dscal(N, DA, DX, INCX) dscal_(N, DA, DX, INCX)
#endif

#ifndef dnrm2
double dnrm2_(int *N, double *X, int *INCX);
#define dnrm2(N, X, INCX) dnrm2_(N, X, INCX)
#endif


/* Aliasing for compiler optimizations */
#ifndef restrict
#ifdef __restrict
#define restrict __restrict
#else
#define restrict
#endif
#endif

/* In-lining for compiler optimizations */
#ifndef inline
#ifdef __inline
#define inline __inline
#else
#define inline 
#endif
#endif

/* RAND() is thread-safe on Windows, but not on *nix */
#ifdef _MSC_VER
#define rand_r(a) rand()
#endif

/* Visual Studio as of 2018 is stuck with OpenMP 2.0 (released 2002),
   which doesn't support parallel loops with unsigned iterators.
   As the code is wrapped in Cython and Cython does not support typdefs conditional on compiler,
   this will map size_t to long on Windows regardless of compiler.
   Can be safely removed if not compiling with MSVC. */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#define size_t long
#else
#include <stddef.h>
#endif

/* Helper functions */
inline int randint(int nmax, unsigned int *seed)
{
	int n = rand_r(seed);
	int lim = INT_MAX - nmax + 1;
	while (n > lim){n = rand_r(seed);}
	return n % nmax;
}

int comp_size_t(const void *a, const void *b)
{
	return ( *(size_t*)a - *(size_t*)b );
}

inline int isin(size_t k, size_t *arr, size_t n)
{
	if (k < arr[0]){return 0;}
	if (k > arr[n-1]){return 0;}
	size_t* res = (size_t*) bsearch(&k, arr, n, sizeof(size_t), comp_size_t);
	return res != NULL;
}

/* Function that applies subgradient updates */
inline void apply_subgradient(double *step_sz, double *buffer_B, double *Anew, double *A, double *B, size_t ia, size_t ib,
	size_t st_buffer_B, size_t k, int k_int, int one, size_t *Acnt, size_t st_buffer_B_cnt, double class)
{
	double res = ddot(&k_int, A + ia*k, &one, B + ib*k, &one);
	if ( ((class == 1) & (res < 1)) || ((class == -1) & (res > -1)) ){
		daxpy(&k_int, step_sz, B + ib*k, &one, Anew + ia*k, &one);
		daxpy(&k_int, step_sz, A + ia*k, &one, buffer_B + st_buffer_B + ib*k, &one);
		Acnt[ia]++;
		buffer_B[st_buffer_B_cnt + ib]++;
	}
}

/* Main function
	A                       :  Already-initialized A matrix (model parameters)
	B                       :  Already-initialized B matrix (model parameters)
	dimA                    :  Number of rows in matrix A
	dimB                    :  Number of rows in matrix B
	k                       :  Dimensionality of low-rank approximation (number of columns in A and B)
	nnz                     :  Number of non-zero entries in the X matrix
	X_indptr, X_ind,  Xr  :  X matrix (dim A x B) in row-sparse format - values indicate weights, Xr is ignored when there's no weights
	reg_param               :  Strength of l2 regularization
	niter                   :  Number of sub-gradient iterations
	projected               :  Whether to apply a projection step at each update (recommended)
	nthreads                :  Number of parallel threads to use
	buffer_B                :  Working memory of dimension <dimB * (k + 1) * nthreads>
	Anew                    :  Working memory of dimension <dimA * k>
	Bnew                    :  Working memory of dimension <dimB * k>
	Acnt                    :  Working memory of dimension <dimA>
	Bcnt                    :  Working memory of dimension <dimB>
*/
void psgd(double *restrict A, double *restrict B, size_t dimA, size_t dimB, size_t k, size_t nnz,
	size_t *restrict X_indptr, size_t *restrict X_ind, double *restrict Xr,
	double reg_param, size_t niter, int projected, int nthreads,
	double *restrict buffer_B, double *restrict Anew, double *restrict Bnew, size_t *restrict Acnt, size_t *restrict Bcnt)
{

	size_t ib;
	int k_int = (int) k;
	double scaling_iter, scaling_mispred, scaling_proj;
	double scaling_proj0 = 1 / sqrt(reg_param);
	double cnst;

	int one = 1;
	double one_dbl = 1;
	double minus_one = -1;

	size_t nthis;
	size_t st_this;
	size_t i;
	int tid;

	size_t dim_bufferB = dimB * (k + 1) * nthreads;
	size_t st_buffer_B;
	size_t st_cnt_buffer = dimB * k * nthreads;
	size_t st_buffer_B_cnt;

	/* Setting different random seeds for each thread
	   Note: MSVC does not support C99 standard, hence this code*/
	#ifdef _MSC_VER
	unsigned int *seeds = (unsigned int *) malloc(sizeof(int) * nthreads);
	#else
	unsigned int seeds[nthreads];
	#endif
	for (int tid = 0; tid < nthreads; tid++){
		seeds[tid] = tid + 1;
	}
	unsigned int* tr_seed;

	/* Iterations of the loop */
	for (size_t t = 1; t < (niter+1) ; t++){

		/* Setting all temporary arrays to zero */
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(Anew) firstprivate(dimA, k)
		for (size_t n = 0; n < (dimA*k); n++){Anew[n] = 0;}
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(Bnew) firstprivate(dimB, k)
		for (size_t n = 0; n < (dimB*k); n++){Bnew[n] = 0;}
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(Acnt) firstprivate(dimA)
		for (size_t n = 0; n < (dimA); n++){Acnt[n] = 0;}
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(Bcnt) firstprivate(dimB)
		for (size_t n = 0; n < (dimB); n++){Bcnt[n] = 0;}
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(buffer_B) firstprivate(dim_bufferB)
		for (size_t n = 0; n < dim_bufferB; n++){buffer_B[n] = 0;}

		/* Scaling parameters for this iteration */
		scaling_iter = 1 - 1 / (double) t;
		scaling_mispred = 1 / (reg_param * (double) t);


		/* 
		Note: the loop here could be done more efficiently with a reduction on {Bnew, Bcnt}, but by default OpenMP won't allocate large arrays
		and will segfault when B is large, hence the temporary variable buffer_B which is defined as shared, without a reduction.
		*/

		/* Calculating sub-gradients - iteration is through the rows of A */
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) firstprivate(X_indptr, X_ind, Xr, A, B, k, k_int, one, st_cnt_buffer, seeds) private(ib, nthis, st_this, tid, st_buffer_B, st_buffer_B_cnt, tr_seed) shared(Anew, Acnt, buffer_B)
		for (size_t ia = 0; ia < dimA; ia++){
			st_this = X_indptr[ia];
			nthis = X_indptr[ia + 1] - st_this;
			tid = omp_get_thread_num();
			st_buffer_B = (dimB * k) * tid;
			st_buffer_B_cnt = st_cnt_buffer + dimB * tid;

			/* Regular case: this row has few entries, can subsample entries at random */
			if (nthis < dimB*0.1){

				/* Sub-gradient for non-zero entries (positive class) */
				for (size_t i = 0; i < nthis; i++){
					size_t ib = X_ind[st_this + i];
					apply_subgradient(Xr + st_this + i, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, one, Acnt, st_buffer_B_cnt, 1);
				}

				/* Sub-gradients for sampled zero entries (negative class) */
				tr_seed = seeds + omp_get_thread_num();
				for (size_t i = 0; i < nthis; i++){
					ib = (size_t) randint(dimB, tr_seed);
					while (isin(ib, X_ind + st_this, nthis)){
						ib = (size_t) randint(dimB, tr_seed);
					}
					apply_subgradient(&minus_one, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, one, Acnt, st_buffer_B_cnt, -1);
				}
			} else {
				/* If this row has too many entries, it will be too slow to subsample entries
				   at random, as there's a look-up for each of them and the probability that they will
				   be present and a new one has to be sampled again will be quite high.

				   In this case, better iterate through all the entries at once */
				for (size_t ib = 0; ib < dimB; ib++){
					i = 0;
					/* Entry is non-zero */
					if (isin(ib, X_ind + st_this, nthis)){
						apply_subgradient(Xr + st_this + i, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, one, Acnt, st_buffer_B_cnt, 1);
						i++;
					/* Entry is zero */
					} else {
						apply_subgradient(&minus_one, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, one, Acnt, st_buffer_B_cnt, -1);
					}
				}
			}
			
		}

		/* Reconstructing Bnew and Bcnt, same as they are for A */
		if (nthreads > 1){
			#pragma omp parallel for schedule(static) num_threads(nthreads) firstprivate(buffer_B, k, one, one_dbl, st_cnt_buffer, dimB) shared(Bnew, Bcnt) private(st_buffer_B_cnt)
			for (size_t ib = 0; ib < dimB; ib++){
				for (int tr = 0; tr < nthreads; tr++){
					st_buffer_B_cnt = st_cnt_buffer + dimB * tr;
					daxpy(&k_int, &one_dbl, buffer_B + tr*(dimB * k) + ib*k, &one, Bnew + ib*k, &one);
					Bcnt[ib] += buffer_B[st_buffer_B_cnt + ib];
				}
			}
		} else {
			Bnew = buffer_B;
			for (size_t ib = 0; ib < dimB; ib++){
				Bcnt[ib] = buffer_B[st_cnt_buffer + ib];
			}
		}

		/* Applying the updates */
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(A) firstprivate(Anew, projected, scaling_mispred, scaling_proj0, Acnt, k, k_int, one) private(cnst, scaling_proj)
		for (size_t ia = 0; ia < dimA; ia++){
			dscal(&k_int, &scaling_iter, A + ia*k, &one);
			if (Acnt[ia] > 0){
				cnst = scaling_mispred / (double) Acnt[ia];
				daxpy(&k_int, &cnst, Anew + ia*k, &one, A + ia*k, &one);
			}
			if (projected){
				scaling_proj = scaling_proj0 / dnrm2(&k_int, A + ia*k, &one);
				if (scaling_proj < 1){dscal(&k_int, &scaling_proj, A + ia*k, &one);}
			}
		}

		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(B) firstprivate(Bnew, projected, scaling_mispred, scaling_proj0, Bcnt, k, k_int, one) private(cnst, scaling_proj)
		for (size_t ib = 0; ib < dimB; ib++){
			dscal(&k_int, &scaling_iter, B + ib*k, &one);
			if (Bcnt[ib] > 0){
				cnst = scaling_mispred / (double) Bcnt[ib];
				daxpy(&k_int, &cnst, Bnew + ib*k, &one, B + ib*k, &one);
			}
			if (projected){
				scaling_proj = scaling_proj0 / dnrm2(&k_int, B + ib*k, &one);
				if (scaling_proj < 1){dscal(&k_int, &scaling_proj, B + ib*k, &one);}
			}
		}
	}

	#ifdef _MSC_VER
	free(seeds);
	#endif

}
