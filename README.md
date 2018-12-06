# Binary Matrix Factorization

This package performs low-rank factorization of sparse binary matrices. Model is based on minimization of hinge loss, and is fit through projected sub-gradient descent updates similar to the _pegasos_ algorithm for SVM. At each iteration, it samples entries from the negative class (having a value of zero) at random with replacement, until having an equal number of missing and non-missing entries for each row in the original matrix.

Package is written in C with a Python interface. Requires a C compiler and some BLAS library such as MKL or OpenBLAS (will try to use the same one that NumPy installation is using).

Computations are parallelized and speed (depending on sparsity) is in the same scale as `implicit-ALS` with the CG method as implemented in the [implicit](https://github.com/benfred/implicit) package.

# Installation

Clone or download the repository and then install with setup.py, e.g.:

```
git clone https://github.com/david-cortes/binmf.git
cd binmf
python setup.py install
```

# Usage

Package does not have a fully-fledged API, only a function that modifies already-initialized parameter matrices in-place.

Takes as input a sparse matrix in CSR or CSC format, and already-initialized factor matrices.

Will take the values in the non-zero entries as weights. If a uniform weighting is desired, the input matrix should have all values equal to 1.

Example:
```python
import numpy as np
from scipy.sparse import csr_matrix

## Generating random sparse data
nrows = 100
ncols = 50
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
```


# References
* Pan, Rong, et al. "One-class collaborative filtering." Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. IEEE, 2008.
* Shalev-Shwartz, Shai, et al. "Pegasos: Primal estimated sub-gradient solver for svm." Mathematical programming 127.1 (2011): 3-30.
