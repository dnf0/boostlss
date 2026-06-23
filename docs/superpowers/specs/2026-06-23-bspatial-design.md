# Bivariate Tensor-Product P-splines (`bspatial`) Design Spec

- Status: Approved
- Date: 2026-06-23

## 1. Goal
Implement a spatial base learner (`BivariatePSpline` / `bspatial`) for 2D spatial coordinates (e.g., latitude and longitude) within the `boostlss` library. It will generate bivariate tensor-product P-splines that are compatible with the existing dense `LearnerFit` solver and boosting engine.

## 2. Architecture

### 2.1 Component Structure
- **Location:** `crates/boostlss/src/learner/bspatial.rs`
- **Struct Definition:** `BivariatePSpline`
  - Needs two column names: `feature1` and `feature2`.
  - Hyperparameters: `knots` (default 20), `degree` (default 3), `differences` (default 2), `df` (default 4.0).
  - Internally, it instantiates two 1D `PSpline` structs, mapping to `feature1` and `feature2` respectively, to handle boundary knots and marginal Cox-de Boor evaluation.

### 2.2 Design Matrix ($B$)
1. **Marginal Bases:** Evaluate marginal design matrices $B_1$ and $B_2$ for the two features using the existing 1D `PSpline::build_design`. This automatically covers boundary extrapolation requirements.
2. **Tensor Product:** Combine marginal bases via a row-wise Kronecker product.
   - For each observation $i$, the basis row is $B_{1,i} \otimes B_{2,i}$.
   - Let $p = \text{knots} + \text{degree} + 1$. The resulting design matrix $B$ has dimensions $N \times (p^2)$.

### 2.3 Penalty Matrix ($K$)
The bivariate penalty matrix is a Kronecker sum of the marginal penalty matrices $K_1$ and $K_2$:
- $K = K_1 \otimes I_{p} + I_{p} \otimes K_2$
- A `kronecker_product(A, B)` utility function will be created in `crates/boostlss/src/learner/util.rs` (or similar utility module) to cleanly generate these sparse-like matrices in dense representation.

### 2.4 Engine Integration
- Add `BivariatePSpline(BivariatePSpline)` to the `BaseLearner` enum in `crates/boostlss/src/learner/mod.rs`.
- Ensure it properly returns dense `ndarray::Array2` structures, allowing seamless compatibility with the `faer` Cholesky factorization in `LearnerFit`.

### 2.5 Python Bindings
- **Location:** `crates/boostlss-py/src/learner.rs`
- Define `PyBivariatePSplineLearner` using `pyclass`.
- Expose the class in the module inside `crates/boostlss-py/src/lib.rs`.
- Add test coverage in `crates/boostlss-py/tests/` to verify instantiation and prediction.

## 3. Scope & Constraints
- **Dense Matrices:** We use dense representations ($p^2 \times p^2$) since typical configurations (e.g., $20 \times 20$ inner knots -> $p=24$, matrix size $576 \times 576$) are computationally cheap for dense Cholesky factorization. Very high-resolution spatial smoothing (e.g., $100 \times 100$ knots) would necessitate sparse solvers, which is deferred to a future roadmap item.
- **Isotropic Smoothing:** This v1 design assumes isotropic smoothing parameters (the same `df` applied to both dimensions), which translates to a single multiplier $\lambda$ for the combined penalty $K$. Anisotropic smoothing is deferred.

## 4. Testing Strategy
- Unit test to assert Kronecker sum properties of the penalty matrix.
- Unit test ensuring row-wise Kronecker basis outputs match expected values based on 1D bases.
- Python unit test establishing boundary consistency and basic predictability.
