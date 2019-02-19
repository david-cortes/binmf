#' @title Binary Low-Rank Factorization for Sparse Matrices
#' @description Creates a low-rank factorization of a sparse binary matrix by
#' minimizing hinge loss, using an similar routine to the _Pegasos_ algorithm
#' for SVM. It subsamples missing entries at random during each iteration by taking as many
#' negative samples (zero-valued) as there are non-missing ones for each row.
#' @param X The matrix to factorize. Can be: a) a `data.frame` with 3 columns, containing in this order:
#' row index, column index, weight; b) A sparse matrix in CSR format from the `SparseM` package;
#' c) a full matrix (of class `matrix` or `Matrix::dgTMatrix`), where zero entries
#' are to be represented as zero and non-missing entries as a positive number indicating their weight;
#' d) a sparse matrix from package `Matrix` in either triplets or CSC formats (will be cast to CSR so triplets
#' is more efficient), with the values being the weights. For uniform weights pass all ones as values in the sparse matrix.
#' @param k Dimensionality of the factorization (a.k.a. number of latent factors).
#' @param reg_param Strength of the l2 regularization.
#' @param niter Number of iterations to run.
#' @param nthreads Number of parallel threads to use.
#' @param projected Whether to apply a projection step (recommended) or not.
#' @references Shalev-Shwartz, Shai, et al. "Pegasos: Primal estimated sub-gradient solver for svm." Mathematical programming 127.1 (2011): 3-30.
#' @examples 
#' nrow <- 10 ** 2
#' ncol <- 10 ** 3
#' nnz <- 10 ** 4
#' X <- data.frame(
#'     row_ix = as.integer(runif(nnz, min = 1, max = nrow)),
#'     col_ix = as.integer(runif(nnz, min = 1, max = ncol)),
#'     weight = 1)
#' X <- X[!duplicated(X[, c("row_ix", "col_ix")]), ]
#' model <- binmf(X)
#' predict(model, 1, 10) ## predict entry (1, 10)
#' predict(model, c(1, 1, 1), c(4, 5, 6)) ## predict entries [1,4], [1,5], [1,6]
#' head(predict(model, 1)) ## predict the whole row 1
#' @export
binmf <- function(X, k = 50, reg_param = 1e-1, niter = 100, nthreads = -1, projected = TRUE) {
	
	### Check input parameters
	if (NROW(niter) > 1 || niter < 1) { stop("'niter' must be a positive integer.") }
	if (NROW(reg_param) > 1 || reg_param < 0) { stop("'reg_param' must be a non-negative number.") }
	if (NROW(nthreads) > 1 || nthreads < 1) {nthreads <- parallel::detectCores()}
	if (NROW(k) > 1 || k < 1) { stop("'k' must be a positive integer.") }
	k <- as.integer(k)
	reg_param <- as.numeric(reg_param)
	niter <- as.integer(niter)
	nthreads <- as.integer(nthreads)
	projected <- as.integer(as.logical(projected))
	
	is_non_int <- FALSE
	
	### Convert X to CSR if it isn't already
	if ("data.frame" %in% class(X)) {
		
		if (!("integer") %in% class(X[[1]]) & !("numeric" %in% class(X[[1]]))) { is_non_int <- TRUE }
		if (!("integer") %in% class(X[[2]]) & !("numeric" %in% class(X[[2]]))) { is_non_int <- TRUE }
		if (is_non_int) {
			X[[1]] <- factor(X[[1]])
			X[[2]] <- factor(X[[2]])
			levels_A <- levels(X[[1]])
			levels_B <- levels(X[[2]])
			X[[1]] <- as.integer(X[[1]])
			X[[2]] <- as.integer(X[[2]])
		}
		
		ix_row <- as.integer(X[[1]])
		ix_col <- as.integer(X[[2]])
		xflat <- as.numeric(X[[3]])
		
		if (any(is.na(ix_row)) || any(is.na(ix_col)) || any(is.na(xflat))) {
			stop("Input contains missing values.")
		}
		Xcsr <- Matrix::sparseMatrix(i = ix_col, j = ix_row, x = xflat, giveCsparse = TRUE)
	} else if ("dgeMatrix" %in% class(X)) {
		if (any(is.na(X))) { stop("Input contains missing values.") }
		Xcsr <- as(t(X), "sparseMatrix")
	} else if ("dgCMatrix" %in% class(X)) {
		Xcsr <- t(X)
	} else if ("dgTMatrix" %in% class(X)) {
		Xcsr <- as(t(X), "sparseMatrix")
	} else if ("matrix" %in% class(X)) {
		if (any(is.na(X))) { stop("Input contains missing values.") }
		Xcsr <- as(t(X), "sparseMatrix")
	} else if ("matrix.csr" %in% class(X)) {
		Xcsr <- X
	} else {
		stop("'X' must be a 'data.frame' with 3 columns, or a matrix (either full or sparse in triplets or compressed).")
	}
	
	### Get dimensions
	dimA <- NCOL(Xcsr)
	dimB <- NROW(Xcsr)
	if ("matrix.csr" %in% class(Xcsr)) {
		nnz <- length(Xcsr@ra)
	} else {
		nnz <- length(Xcsr@x)
	}
	if (nnz < 1) { stop("Input does not contain non-zero values.") }
	
	### Initialize factor matrices
	A = rnorm(dimA * k)
	B = rnorm(dimB * k)
	
	### Run optimizer
	if ("matrix.csr" %in% class(Xcsr)) {
		r_wrapper_binmf(A, B, dimA, dimB, k,
						Xcsr@ra, Xcsr@ja - 1, Xcsr@ia - 1, nnz,
						reg_param, niter, projected, nthreads)
	} else {
		r_wrapper_binmf(A, B, dimA, dimB, k,
						Xcsr@x, Xcsr@i, Xcsr@p, nnz,
						reg_param, niter, projected, nthreads)
	}
	
	### Return all info
	out <- list(
		A = matrix(A, nrow = k, ncol = dimA),
		B = matrix(B, nrow = k, ncol = dimB),
		k = k,
		reg_param = reg_param,
		niter = niter,
		projected = projected,
		dimA = dimA,
		dimB = dimB,
		nnz = nnz,
		nthreads = nthreads
	)
	if (is_non_int) {
		out[["levels_A"]] <- levels_A
		out[["levels_B"]] <- levles_B
	}
	return(structure(out, class = "binmf"))
}

#' @title Make predictions for arbitrary entries in matrix
#' @param object An object of class "binmf" as returned by function "binmf".
#' @param a Row(s) for which to predict.
#' @param b Column(s) for which to predict. If NULL, will make predictions for all columns. Otherwise,
#' it must be of the same length as "a", and the output will contain the prediction for each combination
#' of "a" and "b" passed here.
#' @param ... Extra arguments (not used).
#' @seealso \link{binmf}
#' @export
predict.binmf <- function(object, a, b = NULL, ...) {
	if (!is.null(b) && length(a) != length(b)) { stop("'a' and 'b' must be of the same length.") }
	if ("levels_A" %in% names(object)) {
		a <- as.integer(factor(a, levels = object$levels_A))
		if (!is.null(b)) { b <- as.integer(factor(b, levels = object$levels_B)) }
	} else {
		a <- as.integer(a)
		if (min(a) < 1) { stop("'a' and 'b' must be row/column indexes.") }
		if (max(a) > object$dimA) { stop("Can only make predictions for the same rows and columns from the training data.") }
		if (!is.null(b)) {
			b <- as.integer(b)
			if (min(b) < 1) { stop("'a' and 'b' must be row/column indexes.") }
			if (max(b) > object$dimB) { stop("Can only make predictions for the same rows and columns from the training data.") }
		}
	}
	
	if (is.null(b)) {
		pred <- object$A[, a] %*% object$B
	} else {
		pred = vector(mode = "numeric", length = length(a))
		predict_multiple(object$A, object$B, object$k, length(a), a, b, pred, object$nthreads)
	}
	pred <- as.vector(pred)
	
	if (!is.null(b) && "levels_A" %in% names(object)) {
		names(pred) <- object$levels_B
		return(pred)
	} else {
		return(pred)
	}
}

#' @title Get information about binmf object
#' @description Print basic properties of a "binmf" object.
#' @param x An object of class "binmf" as returned by function "binmf".
#' @param ... Extra arguments (not used).
#' @export
print.binmf <- function(x, ...) {
	cat("Binary Matrix Factorization\n\n")
	cat("Number of rows:", x$dimA, "\n")
	cat("Number of columns:", x$dimB, "\n")
	cat("Number of non-zero entries:", x$nnz, "\n")
	cat("Dimensionality of factorization:", x$k, "\n")
	cat("Regularization parameter:", x$reg_param, "\n")
	
	if ("levels_A" %in% names(x)) {
		cat("\nRow names:", head(x$levels_A))
		cat("\nColumn names:", head(x$levels_B), "\n")
	}
}

#' @title Get information about binmf object
#' @description Print basic properties of a "binmf" object (same as `print.binmf` function).
#' @param x An object of class "binmf" as returned by function "binmf".
#' @param ... Extra arguments (not used).
#' @seealso \link{print.binmf}
#' @export
summary.binmf <- function(object, ...) {
	print.binmf(object)
}
