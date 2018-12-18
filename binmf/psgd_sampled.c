 /* Sparse Binary Matrix Factorization

 Fit through projected sub-gradient descent, sampling missing entries at random in each iteration.
 Writen for C99 standard.

 Copyright David Cortes 2018 */

#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include "findblas.h" /* https://github.com/david-cortes/findblas */
#ifdef _OPENMP
	#include <omp.h>
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
#ifndef rand_r
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
inline void apply_subgradient(double step_sz, double *buffer_B, double *Anew, double *A, double *B, size_t ia, size_t ib,
	size_t st_buffer_B, size_t k, int k_int, size_t *Acnt, size_t *buffer_B_cnt, size_t st_buffer_B_cnt, double class)
{
	double res = cblas_ddot(k_int, A + ia*k, 1, B + ib*k, 1);
	if ( ((class == 1) & (res < 1)) || ((class == -1) & (res > -1)) ){
		cblas_daxpy(k_int, step_sz, B + ib*k, 1, Anew + ia*k, 1);
		cblas_daxpy(k_int, step_sz, A + ia*k, 1, buffer_B + st_buffer_B + ib*k, 1);
		Acnt[ia]++;
		buffer_B_cnt[st_buffer_B_cnt + ib]++;
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
*/
void psgd(double *restrict A, double *restrict B, size_t dimA, size_t dimB, size_t k, size_t nnz,
	size_t *restrict X_indptr, size_t *restrict X_ind, double *restrict Xr,
	double reg_param, size_t niter, int projected, int nthreads)
{

	size_t ib;
	int k_int = (int) k;
	double scaling_iter, scaling_mispred, scaling_proj;
	double scaling_proj0 = 1 / sqrt(reg_param);
	double cnst;

	size_t nthis;
	size_t st_this;
	size_t i;
	int tid;

	#ifdef _OPENMP
	/* Setting different random seeds for each thread
	   Note: MSVC does not support C99 standard, hence this code*/
		#ifdef _MSC_VER
			unsigned int *seeds = (unsigned int*) malloc(sizeof(int) * nthreads);
		#else
			unsigned int seeds[nthreads];
		#endif
		for (int tid = 0; tid < nthreads; tid++){seeds[tid] = tid + 1;}
	#else
		tid = 0;
		nthreads = 1;
	#endif
	unsigned int* tr_seed;

	double *Anew = (double*) malloc(sizeof(double) * dimA * k);
	double *Bnew = (double*) malloc(sizeof(double) * dimB * k);
	size_t *Acnt = (size_t*) malloc(sizeof(size_t) * dimA);
	size_t *Bcnt = (size_t*) malloc(sizeof(size_t) * dimB);

	
	/*	The idea for these is to create a copy of Bnew and Bcnt for each thread,
		but as OpenMP allocates private arrays in the stack and these can get very big,
		using them as 'private' or 'firstprivate' will instead segfault.
		The code here allocates continuous arrays that are a multiple 'nthreads' of the
		number of elements in Bnew and Bcnt, then each thread writes to its own chunk of these
		arrays to avoid simultaneous edits, and later these are combined into Bnew and Bcnt */
	double *buffer_B;
	size_t *buffer_B_cnt;
	if (nthreads > 1){
		buffer_B = (double*) malloc(sizeof(double) * dimB * k * nthreads);
		buffer_B_cnt = (size_t*) malloc(sizeof(size_t) * dimB * nthreads);
	} else {
		buffer_B = Bnew;
		buffer_B_cnt = Bcnt;
	}

	size_t dim_bufferB = dimB * k * nthreads;
	size_t dim_bufferB_cnt = dimB * nthreads;
	size_t st_buffer_B;
	size_t st_buffer_B_cnt;

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
		#pragma omp parallel for schedule(static) num_threads(nthreads) shared(buffer_B_cnt) firstprivate(dim_bufferB_cnt)
		for (size_t n = 0; n < dim_bufferB_cnt; n++){buffer_B_cnt[n] = 0;}

		/* Scaling parameters for this iteration */
		scaling_iter = 1 - 1 / (double) t;
		scaling_mispred = 1 / (reg_param * (double) t);


		/* 
		Note: the loop here could be done more efficiently with a reduction on {Bnew, Bcnt}, but by default OpenMP won't allocate large arrays
		and will segfault when B is large, hence the temporary variable buffer_B which is defined as shared, without a reduction.
		*/

		/* Calculating sub-gradients - iteration is through the rows of A */
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) firstprivate(X_indptr, X_ind, Xr, A, B, k, k_int, seeds) private(ib, nthis, st_this, tid, st_buffer_B, st_buffer_B_cnt, tr_seed, i) shared(Anew, Acnt, buffer_B, buffer_B_cnt)
		for (size_t ia = 0; ia < dimA; ia++){
			st_this = X_indptr[ia];
			nthis = X_indptr[ia + 1] - st_this;
			#ifdef _OPENMP
			tid = omp_get_thread_num();
			#endif
			st_buffer_B = (dimB * k) * tid;
			st_buffer_B_cnt = dimB * tid;

			/* Regular case: this row has few entries, can subsample entries at random fast */
			if (nthis < dimB*0.1){

				/* Sub-gradient for non-zero entries (positive class) */
				for (size_t i = 0; i < nthis; i++){
					size_t ib = X_ind[st_this + i];
					apply_subgradient(Xr[st_this + i], buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, 1);
				}

				/* Sub-gradients for sampled zero entries (negative class) */
				#ifdef _OPENMP
				tr_seed = seeds + omp_get_thread_num();
				#else
				*tr_seed = 1;
				#endif
				for (size_t i = 0; i < nthis; i++){
					ib = (size_t) randint(dimB, tr_seed);
					while (isin(ib, X_ind + st_this, nthis)){
						ib = (size_t) randint(dimB, tr_seed);
					}
					apply_subgradient(-1, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, -1);
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
						apply_subgradient(Xr[st_this + i], buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, 1);
						i++;
					/* Entry is zero */
					} else {
						apply_subgradient(-1, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, -1);
					}
				}
			}
			
		}

		/* Reconstructing Bnew and Bcnt, same as they are for A */
		if (nthreads > 1){
			#pragma omp parallel for schedule(static) num_threads(nthreads) firstprivate(buffer_B, buffer_B_cnt, k, dimB) shared(Bnew, Bcnt)
			for (size_t ib = 0; ib < dimB; ib++){
				for (int tr = 0; tr < nthreads; tr++){
					cblas_daxpy(k_int, 1, buffer_B + tr*(dimB * k) + ib*k, 1, Bnew + ib*k, 1);
					Bcnt[ib] += buffer_B_cnt[dimB*tr + ib];
				}
			}
		}

		/* Applying the updates */
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(A) firstprivate(Anew, projected, scaling_mispred, scaling_proj0, Acnt, k, k_int) private(cnst, scaling_proj)
		for (size_t ia = 0; ia < dimA; ia++){
			cblas_dscal(k_int, scaling_iter, A + ia*k, 1);
			if (Acnt[ia] > 0){
				cnst = scaling_mispred / (double) Acnt[ia];
				cblas_daxpy(k_int, cnst, Anew + ia*k, 1, A + ia*k, 1);
			}
			if (projected){
				scaling_proj = scaling_proj0 / cblas_dnrm2(k_int, A + ia*k, 1);
				if (scaling_proj < 1){cblas_dscal(k_int, scaling_proj, A + ia*k, 1);}
			}
		}

		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(B) firstprivate(Bnew, projected, scaling_mispred, scaling_proj0, Bcnt, k, k_int) private(cnst, scaling_proj)
		for (size_t ib = 0; ib < dimB; ib++){
			cblas_dscal(k_int, scaling_iter, B + ib*k, 1);
			if (Bcnt[ib] > 0){
				cnst = scaling_mispred / (double) Bcnt[ib];
				cblas_daxpy(k_int, cnst, Bnew + ib*k, 1, B + ib*k, 1);
			}
			if (projected){
				scaling_proj = scaling_proj0 / cblas_dnrm2(k_int, B + ib*k, 1);
				if (scaling_proj < 1){cblas_dscal(k_int, scaling_proj, B + ib*k, 1);}
			}
		}
	}

	if (nthreads > 1){
		free(buffer_B);
		free(buffer_B_cnt);
	}
	free(Anew);
	free(Bnew);
	free(Acnt);
	free(Bcnt);

	#ifdef _OPENMP
		#ifdef _MSC_VER
			free(seeds);
		#endif
	#endif

}
