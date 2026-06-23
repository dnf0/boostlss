# Cyclic P-Splines Design

## Overview
Implement Cyclic Penalized Splines (P-splines) to model periodic effects (e.g., hours of the day, months of the year, seasonal trends). The implementation will extend the existing `PSpline` learner rather than creating a new struct, as the basis evaluation logic is identical except for matrix wrapping at the boundaries.

## Architecture & API Updates

### 1. Rust API
- Add `pub is_cyclic: bool` field to `crates/boostlss/src/learner/pspline.rs` defaulting to `false`.
- Add a builder method `pub fn cyclic(mut self, cyclic: bool) -> Self`.

### 2. Python API
- Modify `PyPSplineLearner` in `crates/boostlss-py/src/model.rs` to accept an optional `cyclic: bool = False` argument in the PyO3 signature.
- Pass this boolean down to the Rust struct during learner extraction.

## Internal Logic Modifications

### 1. Basis Wrapping
For cyclic splines, the function values and derivatives must match exactly at the boundaries.
When `is_cyclic == true`:
- The standard B-spline matrix `B` is built with dimension `(n, knots + degree + 1)`.
- The rightmost `degree` columns are added to the leftmost `degree` columns.
- The matrix is then truncated, dropping the rightmost `degree` columns.
- The returned shape will be `(n, knots + 1)`.

### 2. Cyclic Penalty Matrix
The penalty matrix must enforce smoothness across the boundary (the last coefficient is penalized against the first coefficient).
- Modify `penalty_matrix(n_cols: usize, differences: usize)` in `crates/boostlss/src/learner/penalty.rs` to take `is_cyclic: bool`.
- If `is_cyclic == false`, it behaves exactly as it currently does.
- If `is_cyclic == true`, the difference operator must "wrap around". For a first-order difference (`d=1`), row `n-1` will compute the difference between `c_{n-1}` and `c_0`. Higher-order differences wrap accordingly.

## Testing Strategy
- **Rust Unit Tests:**
  - Test `pspline.cyclic(true)` creates correct dimensions.
  - Test cyclic penalty matrix properly wraps difference values.
- **Python Integration:** Test fitting a periodic dataset (e.g., a sine wave mapping 0..2pi) with a cyclic spline and ensure continuity at the boundary (prediction at 0 equals prediction at 2pi).
