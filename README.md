# Binary Matrix Factorization

This package performs low-rank factorization of sparse binary matrices. Model is based on minimization of hinge loss, and is fit through projected sub-gradient descent updates similar to the _pegasos_ algorithm for SVM. At each iteration, it samples entries from the negative class (having a value of zero) at random with replacement, until having an equal number of missing and non-missing entries for each row in the original matrix.

Package is written in C with Python and R interfaces. Requires a C compiler and some BLAS library such as MKL or OpenBLAS (in Python, will try to use the same one that NumPy installation is using).

Computations are parallelized and speed (depending on sparsity) is in the same scale as `implicit-ALS` with the CG method as implemented in the [implicit](https://github.com/benfred/implicit) package.

# Installation

* Python
Clone or download the repository and then install with setup.py, e.g.:

```
git clone https://github.com/david-cortes/binmf.git
cd binmf
python setup.py install
```
(It requires package `findblas`, can be installed with `pip install findblas`.)

* R
```r
devtools::install_github("david-cortes/binmf")
```

# Usage

* R
See documentation (`help(binmf::binmf`)

* Python

Package does not have a fully-fledged API, only a function that modifies already-initialized parameter matrices in-place.

Takes as input a sparse matrix in CSR format, and already-initialized factor matrices.

Will take the values in the non-zero entries as weights. If a uniform weighting is desired, the input matrix should have all values equal to 1.

Example:
```python
import numpy as np
from scipy.sparse import csr_matrix

## Generating random sparse data
nrows = 100
ncols = 500 ## will be a lot slower when rows are not too sparse
sparsity = 0.1
np.random.seed(1)
dense_mat = (np.random.random(size=(nrows, ncols)) >= sparsity).astype('uint8')
sp_mat = csr_matrix(dense_mat)

## Initializing factor matrices
k = 5
np.random.seed(2)
row_fact = np.random.normal(size=(nrows, k))
col_fact = np.random.normal(size=(ncols, k))

from binmf import fit_spfact
fit_spfact(sp_mat, row_fact, col_fact, reg_param=1e-1, niter=100, nthreads=1) #adjust number of threads for your setup

## Prediction for entry (10, 15)
np.dot(row_fact[10], col_fact[15]) # If greater than zero, prediction is value 1, otherwise prediction is value 0

## Note: more iterations will do better in very large datasets
```

Can also be called directly from C and wrapped in other languages:

```c
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
void psgd(double *A, double *B, size_t dimA, size_t dimB, size_t k, size_t nnz,
	size_t *X_indptr, size_t *X_ind, double *Xr,
	double reg_param, size_t niter, int projected, int nthreads)
```


# References
* Pan, Rong, et al. "One-class collaborative filtering." Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. IEEE, 2008.
* Shalev-Shwartz, Shai, et al. "Pegasos: Primal estimated sub-gradient solver for svm." Mathematical programming 127.1 (2011): 3-30.
