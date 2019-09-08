 /*
	Sparse Binary Matrix Factorization

	Fit through projected sub-gradient descent, sampling missing entries at random in each iteration.
	Writen for C99 standard.

	BSD 2-Clause License

	Copyright (c) 2019, David Cortes
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice, this
	  list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above copyright notice,
	  this list of conditions and the following disclaimer in the documentation
	  and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h> /* memset */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef _FOR_R
	#include "findblas.h" /* https://github.com/david-cortes/findblas */
#else
	#include <R_ext/BLAS.h>
	double cblas_ddot(int n, double *x, int incx, double *y, int incy) { return ddot_(&n, x, &incx, y, &incy); }
	void cblas_daxpy(int n, double a, double *x, int incx, double *y, int incy) { daxpy_(&n, &a, x, &incx, y, &incy); }
	void cblas_dscal(int n, double alpha, double *x, int incx) { dscal_(&n, &alpha, x, &incx); }
	double cblas_dnrm2(int n, double *x, int incx) { return dnrm2_(&n, x, &incx); }
#endif
#ifndef _FOR_R
	#include <stdio.h>
#else
	#include <R_ext/Print.h>
	#define fprintf(f, message) REprintf(message)
#endif
	#ifdef __cplusplus
}
#endif
#ifdef _OPENMP
	#include <omp.h>
#endif

/* Aliasing for compiler optimizations */
#ifdef __cplusplus
	#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER)
		#define restrict __restrict
	#else
		#define restrict 
	#endif
#elif defined(_MSC_VER)
	#define restrict __restrict
#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
	#define restrict 
#endif

/* In-lining for compiler optimizations */
#ifndef __cplusplus
	#if defined(_MSC_VER)
		#define inline __inline
	#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
		#define inline 
	#endif
#endif

/* RAND() is thread-safe on Windows, but not on *nix */
#ifndef rand_r
	#define rand_r(a) rand()
#endif

/* Visual Studio as of 2018 is stuck with OpenMP 2.0 (released 2002),
   which doesn't support parallel loops with unsigned iterators,
   and doesn't support declaring the iterator count right in the loop.
   As the code is wrapped in Cython and Cython does not support typdefs conditional on compiler,
   this will map size_t to long on Windows regardless of compiler.
   Can be safely removed if not compiling with MSVC. */
#ifdef _OPENMP
	#if (_OPENMP > 200801) && !defined(_WIN32) && !defined(_WIN64) /* OpenMP >= 3.0 */
		#define size_t_for size_t
	#else
		#define size_t_for
	#endif
#else
	#define size_t_for size_t
#endif

/* Helper functions */
inline size_t randint(size_t nmax, unsigned int *seed)
{
	if (nmax <= INT_MAX)
	{
		int lim = INT_MAX - INT_MAX % nmax;
		int n;
		do { n = rand_r(seed); } while (n > lim);
		return n % nmax;
	}

	else if (nmax <= UINT_MAX)
	{
		unsigned int lim = UINT_MAX - UINT_MAX % nmax;
		unsigned int n;
		do
		{
			n  = rand_r(seed);
			n += rand_r(seed);
		} while (n > lim);
		return n % nmax;
	}

	else
	{
		size_t ndraws = sizeof(size_t) / sizeof(int);
		size_t n_remainder = sizeof(size_t) % sizeof(int);

		size_t lim = SIZE_MAX - SIZE_MAX % nmax;
		size_t n;
		char *ptr_drawn = (char*) &n;
		unsigned int single_int;
		do
		{
			for (size_t d = 0; d < ndraws; d++) {
				single_int  = rand_r(seed);
				single_int += rand_r(seed);
				memcpy(ptr_drawn + d * sizeof(int), &single_int, sizeof(int));
			}
			if (n_remainder) {
				single_int  = rand_r(seed);
				single_int += rand_r(seed);
				memcpy(ptr_drawn + ndraws * sizeof(int), &single_int, n_remainder);
			}
		} while (n > lim);
		return n % nmax;
	}
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

inline void set_to_zero_dbl(double arr[], const size_t n, const int nthreads)
{
	#if defined(_OPENMP)

	int i;
	size_t chunk_size = n / nthreads;
	size_t remainder = n % nthreads;

	#pragma omp parallel for schedule(static, 1) firstprivate(arr, chunk_size, nthreads)
	for (i = 0; i < nthreads; i++){
		memset(arr + i * chunk_size, 0, sizeof(double) * chunk_size);
	}
	if (remainder > 0){
		memset(arr + nthreads * chunk_size, 0, sizeof(double) * remainder);
	}

	#else
	memset(arr, 0, sizeof(double) * n);
	#endif
}

inline void set_to_zero_szt(size_t arr[], const size_t n, const int nthreads)
{

	#if defined(_OPENMP)

	int i;
	size_t chunk_size = n / nthreads;
	size_t remainder = n % nthreads;

	#pragma omp parallel for schedule(static, 1) firstprivate(arr, chunk_size, nthreads)
	for (i = 0; i < nthreads; i++){
		memset(arr + i * chunk_size, 0, sizeof(size_t) * chunk_size);
	}
	if (remainder > 0){
		memset(arr + nthreads * chunk_size, 0, sizeof(size_t) * remainder);
	}

	#else
	memset(arr, 0, sizeof(size_t) * n);
	#endif
}

/* Function that applies subgradient updates */
inline void add_subgradient(double step_sz, double *buffer_B, double *Anew, double *A, double *B, size_t ia, size_t ib,
	size_t st_buffer_B, size_t k, int k_int, size_t *restrict Acnt, size_t *restrict buffer_B_cnt, size_t st_buffer_B_cnt, double class)
{
	double res = cblas_ddot(k_int, A + ia*k, 1, B + ib*k, 1);
	if ( ((class == 1) & (res < 1)) || ((class == -1) & (res > -1)) ){
		cblas_daxpy(k_int, step_sz, B + ib*k, 1, Anew + ia*k, 1);
		cblas_daxpy(k_int, step_sz, A + ia*k, 1, buffer_B + st_buffer_B + ib*k, 1);
		Acnt[ia]++;
		buffer_B_cnt[st_buffer_B_cnt + ib]++;
	}
}

/* The names on this function assume it's being applied to matrix A - you can pass matrix B just fine too */
inline void update_weights(double *A, double *Anew, size_t *Acnt, size_t dimA, size_t k, int nthreads, int projected, double scaling_mispred, double scaling_proj0, double scaling_iter)
{
	int k_int = (int) k;
	double cnst, scaling_proj;
	#ifdef _OPENMP
		#if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
			long ia;
		#endif
	#endif
	#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(A) firstprivate(Anew, projected, scaling_mispred, scaling_proj0, Acnt, dimA, k, k_int) private(cnst, scaling_proj)
	for (size_t_for ia = 0; ia < dimA; ia++){
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
}

inline void set_arrays_to_zero(double *restrict Anew, double *restrict Bnew,
	size_t *restrict Acnt, size_t *restrict Bcnt, double *restrict buffer_B, size_t *restrict buffer_B_cnt,
	size_t dimA, size_t dimB, size_t k, size_t dim_bufferB, size_t dim_bufferB_cnt, int nthreads)
{

	set_to_zero_dbl(Anew, dimA * k, nthreads);
	set_to_zero_dbl(Bnew, dimB * k, nthreads);
	set_to_zero_szt(Acnt, dimA, nthreads);
	set_to_zero_szt(Bcnt, dimB, nthreads);
	set_to_zero_dbl(buffer_B, dim_bufferB, nthreads);
	set_to_zero_szt(buffer_B_cnt, dim_bufferB_cnt, nthreads);
}

inline void reconstruct_B_arrays(double *buffer_B, size_t *buffer_B_cnt, double *Bnew, size_t *Bcnt, size_t dimB, size_t k, int nthreads)
{
	
	int k_int = (int) k;

	#ifdef _OPENMP
		#if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
			long ib;
		#endif
	#endif

	#pragma omp parallel for schedule(static, dimB/nthreads) num_threads(nthreads) firstprivate(buffer_B, buffer_B_cnt, k, dimB) shared(Bnew, Bcnt)
	for (size_t_for ib = 0; ib < dimB; ib++){
		for (int tr = 0; tr < nthreads; tr++){
			cblas_daxpy(k_int, 1, buffer_B + tr*(dimB * k) + ib*k, 1, Bnew + ib*k, 1);
			Bcnt[ib] += buffer_B_cnt[dimB*tr + ib];
		}
	}
}


/* Main function
	A                       :  Already-initialized A matrix (model parameters)
	B                       :  Already-initialized B matrix (model parameters)
	dimA                    :  Number of rows in matrix A
	dimB                    :  Number of rows in matrix B
	k                       :  Dimensionality of low-rank approximation (number of columns in A and B)
	nnz                     :  Number of non-zero entries in the X matrix
	X_indptr, X_ind,  Xr    :  X matrix (dim A x B) in row-sparse format - values indicate weights, Xr is ignored when there's no weights
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
	double scaling_iter, scaling_mispred;
	double scaling_proj0 = 1.0 / sqrt(reg_param);

	size_t nthis;
	size_t st_this;
	size_t i;
	int tid;

	/* Avoid nested parallelism */
	#ifdef _OPENMP
		#if defined(_MKL_H_)
			mkl_set_num_threads_local(1);
		#elif defined(CBLAS_H)
			openblas_set_num_threads(1);
		#endif
	#endif

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

	if (Anew == NULL || Bnew == NULL || Acnt == NULL || Bcnt == NULL ||
		buffer_B == NULL || buffer_B_cnt == NULL
		#if defined(_OPENMP)
		|| seeds == NULL
		#endif
		) {fprintf(stderr, "Error: Could not allocate memory for procedure.\n"); goto cleanup;}

	size_t dim_bufferB = dimB * k * nthreads;
	size_t dim_bufferB_cnt = dimB * nthreads;
	size_t st_buffer_B;
	size_t st_buffer_B_cnt;

	/* Iterations of the loop */
	for (size_t t = 1; t < (niter+1) ; t++){

		/* All arrays should be reset */
		set_arrays_to_zero(Anew, Bnew, Acnt, Bcnt, buffer_B, buffer_B_cnt, dimA, dimB, k, dim_bufferB, dim_bufferB_cnt, nthreads);

		/* Scaling parameters for this iteration */
		scaling_iter = 1 - 1 / (double) t;
		scaling_mispred = 1 / (reg_param * (double) t);


		/*  Note: the loop here could be done more efficiently with a reduction on {Bnew, Bcnt}, but by default OpenMP won't allocate large arrays
		and will segfault when B is large, hence the temporary variable buffer_B which is defined as shared, without a reduction. */

		/* Calculating sub-gradients - iteration is through the rows of A */
		#ifdef _OPENMP
			#if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
				long ia;
			#endif
		#endif
		#pragma omp parallel for schedule(dynamic) num_threads(nthreads) firstprivate(X_indptr, X_ind, Xr, A, B, k, k_int, seeds) private(ib, nthis, st_this, tid, st_buffer_B, st_buffer_B_cnt, tr_seed, i) shared(Anew, Acnt, buffer_B, buffer_B_cnt)
		for (size_t_for ia = 0; ia < dimA; ia++){
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
					add_subgradient(Xr[st_this + i], buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, 1);
				}

				/* Sub-gradients for sampled zero entries (negative class) */
				#ifdef _OPENMP
					tr_seed = seeds + omp_get_thread_num();
				#else
					*tr_seed = 1;
				#endif
				for (size_t i = 0; i < nthis; i++){

					/* Pick a random number from B that is not in this row of A */
					do { ib = randint(dimB, tr_seed); } while ( isin(ib, X_ind + st_this, nthis) );
					add_subgradient(-1, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, -1);
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
						add_subgradient(Xr[st_this + i], buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, 1);
						i++;
					/* Entry is zero */
					} else {
						add_subgradient(-1, buffer_B, Anew, A, B, ia, ib, st_buffer_B, k, k_int, Acnt, buffer_B_cnt, st_buffer_B_cnt, -1);
					}
				}
			}
			
		}

		/* Reconstructing Bnew and Bcnt, same as they are for A */
		if (nthreads > 1){reconstruct_B_arrays(buffer_B, buffer_B_cnt, Bnew, Bcnt, dimB, k, nthreads);}

		/* Applying the updates */
		update_weights(A, Anew, Acnt, dimA, k, nthreads, projected, scaling_mispred, scaling_proj0, scaling_iter);
		update_weights(B, Bnew, Bcnt, dimB, k, nthreads, projected, scaling_mispred, scaling_proj0, scaling_iter);

	}

	cleanup:
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
