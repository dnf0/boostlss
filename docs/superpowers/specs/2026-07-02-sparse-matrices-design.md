# Sparse Design Matrices — Design Spec

- Status: Draft for review
- Date: 2026-07-02
- Scope: Upgrading the `boostlss` core engine to support sparse design matrices (CSR/CSC) across all base learners, preventing Out-Of-Memory (OOM) errors on high-dimensional data (e.g., `RandomEffects` with many categories).

---

## 1. Goal

Eliminate the massive memory bottleneck caused by dense design matrices in base learners like `RandomEffects`. Enable the entire boosting hot-path—from design matrix construction to `X^T W X` Gram matrix calculation and Cholesky factorization—to operate entirely in sparse format when applicable.

## 2. Approach: Hybrid `DesignMatrix` Resolution

We will update the core `BaseLearner` trait and the `LearnerFit` caching system to seamlessly handle both `Dense` and `Sparse` mathematical operations, preserving the performance of dense matrices for small models while unlocking infinite scalability for sparse models.

### 2.1 Updating the Data Layer
The `Dataset` struct currently has a `DesignMatrix` enum (`Dense`, `Csr`, `Csc`).
We will propagate this type upwards:
* `BaseLearner::build_design` will change its signature from `Result<Array2<f64>, ...>` to `Result<DesignMatrix, ...>`.
* `BaseLearner::penalty_matrix` will change from `Array2<f64>` to `DesignMatrix`.

### 2.2 Base Learner Implementations
* **RandomEffects**: Will be rewritten to output a `DesignMatrix::Csc` directly. The penalty matrix will be a `DesignMatrix::Csc` (Sparse Identity). This prevents the current O(n_obs * n_categories) memory allocation.
* **Linear**: Will output `DesignMatrix::Csc` or `DesignMatrix::Csr` if the underlying `Dataset` was provided as a sparse matrix. Otherwise, it defaults to `Dense`.
* **PSpline / ConstrainedPSpline**: Will continue to output `Dense` by default for v1 sparse support, with the architecture now fully capable of allowing a sparse upgrade in the future. (B-spline bases are inherently banded/sparse).

### 2.3 The Solver State (`SolverState`)
The `LinearFitState` (which represents the cached Cholesky factorization for a base learner) will be upgraded to an enum to handle both dense and sparse solvers:

```rust
pub enum SolverState {
    Dense {
        coef: Array1<f64>,
        llt: faer::linalg::solvers::Llt<f64>,
        design: Array2<f64>,
    },
    Sparse {
        coef: Array1<f64>,
        // We will utilize a sparse Cholesky solver (e.g., from `sprs` or `faer::sparse`)
        llt: SparseCholeskySolver,
        design: SparseMatrix, // Csr or Csc
    }
}
```

### 2.4 The Boosting Hot-Path
In `initialize` (which occurs once per learner before boosting begins):
1. Compute `X^T W X + lambda * K`.
   * If `X` and `K` are `Dense`, use standard dense matrix multiplication.
   * If `X` and `K` are `Sparse`, use sparse matrix multiplication (e.g., `sprs` crate).
2. Factorize the resulting Gram matrix.
   * If `Dense`, use `faer::linalg::solvers::Llt` (existing).
   * If `Sparse`, use the sparse Cholesky factorization.

In `fit_update` (which occurs every boosting iteration):
1. Compute `X^T * u` (the pseudo-response projection).
   * Handled via Dense or Sparse matrix-vector multiplication.
2. Solve `(X^T W X + lambda * K) * beta = X^T * u` using the cached `llt`.

In `predict_update` (which occurs to update predictions):
1. Compute `X * beta`.
   * Handled via Dense or Sparse matrix-vector multiplication.

## 3. Dependencies
We will add the `sprs` (Sparse Matrices) and `sprs-ldl` (Sparse Cholesky/LDL) crates if `faer::sparse` proves insufficient or unstable for these operations. `sprs` provides robust CSR/CSC formats and matrix multiplications required for `X^T W X`.

## 4. Migration Path
1. Add necessary sparse dependencies.
2. Update `DesignMatrix` enum in `data.rs` to support matrix multiplication and transposes.
3. Update `BaseLearner` trait signatures.
4. Upgrade `RandomEffects` to output sparse matrices.
5. Upgrade `LearnerFit::initialize` to branch on Dense vs. Sparse and compute the appropriate Cholesky factor.
6. Verify all existing tests pass and add an OOM-prevention test for `RandomEffects` with 1,000,000 categories.
