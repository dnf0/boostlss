# Constrained PSpline Design Spec

- Status: Draft for review
- Date: 2026-06-24
- Author: Daniel Fisher (with Antigravity)
- Scope: Implementation of `ConstrainedPSpline` base-learner for monotonic and convex/concave constraints in `boostlss`.

## 1. Goal and Constraints

We need to support shape constraints (monotonic increasing/decreasing, convex/concave) for additive predictors in `boostlss`.

In standard penalized B-splines, the penalty matrix $K$ is fixed (e.g., $D^T D$ where $D$ is the difference matrix).
For constrained P-splines (as implemented in `mboost` / `bmono`), the penalty is dynamically re-weighted based on the coefficient estimates: we only penalize differences that violate the desired constraint.
For example, for a monotonic increasing constraint with $d=1$, we want $\beta_j \ge \beta_{j-1}$. If $\beta_j < \beta_{j-1}$, we heavily penalize the difference.

Because the penalty depends on $\beta$, finding the optimal $\beta$ requires an Iteratively Reweighted Least Squares (IRLS) algorithm inside the base-learner's fit step.

## 2. Architecture

**Approach 1: Standalone `ConstrainedPSpline`**

We will create a new base-learner `ConstrainedPSpline` in `crates/boostlss/src/learner/constrained_pspline.rs`.

- **Constraint Enum:**
  ```rust
  pub enum Constraint {
      MonotonicIncreasing,
      MonotonicDecreasing,
      Convex,
      Concave,
  }
  ```

- **Configuration:**
  - `feature_idx`, `knots`, `degree`, `differences`, `df`.
  - `constraint: Constraint`.
  - `max_iter: usize` (default 10)
  - `tolerance: f64` (default 1e-6)

- **Design Matrix:**
  The B-spline basis generation is mathematically identical to standard `PSpline`.
  We will extract the `PSpline` basis building logic (currently in `pspline.rs:build_design`) into a shared utility `crates/boostlss/src/learner/spline_utils.rs` so both `PSpline` and `ConstrainedPSpline` can use it without duplication.

- **Fitting Logic (IRLS):**
  Instead of a single Cholesky decomposition, `initialize()` will setup the data.
  `fit_update(u, weights)` will:
  1. Start with an unconstrained fit (or the previous $\beta$).
  2. Loop up to `max_iter`:
     - Compute the differences $D \beta$.
     - Compute an asymmetric weight vector $v$. E.g., for `MonotonicIncreasing` ($d=1$), $v_i = 1$ if $(D\beta)_i < 0$ else $0$.
     - Construct the active penalty matrix: $K = \lambda K_{smooth} + \lambda_{cons} D^T \text{diag}(v) D$. Note: $\lambda_{cons}$ is typically very large (e.g. $10^6$) to enforce the constraint.
     - Solve $(X^T W X + K) \beta_{new} = X^T W u$.
     - Check if $\max|\beta_{new} - \beta| < \text{tolerance}$. If so, break.
  3. Return the constrained $\beta$ as a `LearnerUpdate::Linear`.

- **Enum Integration:**
  Add `ConstrainedPSpline` to the `BaseLearner` enum in `learner/mod.rs` and `model.rs`.

## 3. IRLS Details

For `bmono` in `mboost`:
- **Differences:** `differences` parameter for the *smoothing* penalty is usually 1 or 2 (default 2 in `mboost` for smoothing, but 1 for monotonicity). To keep things simple, we'll allow specifying them separately or just use `differences=1` for monotonic and `differences=2` for convex/concave.
- **Asymmetric Penalty:**
  - Let $D_c$ be the difference matrix of order $c$ ($c=1$ for monotonic, $c=2$ for convex/concave).
  - Let $v$ be a vector of size $p-c$.
  - For Monotonic Increasing: $v_i = 1$ if $(D_1 \beta)_i < 0$, else $0$.
  - For Monotonic Decreasing: $v_i = 1$ if $(D_1 \beta)_i > 0$, else $0$.
  - For Convex: $v_i = 1$ if $(D_2 \beta)_i < 0$, else $0$.
  - For Concave: $v_i = 1$ if $(D_2 \beta)_i > 0$, else $0$.
- The total penalty is $P = \lambda_{smooth} D_s^T D_s + \kappa D_c^T \text{diag}(v) D_c$.
  - $\lambda_{smooth}$ is chosen via `df_to_lambda`.
  - $\kappa$ is a large constant (e.g., $10^6$) to strictly enforce the constraint.

## 4. Tests
- Unit tests verifying that the constraints are actually respected on toy data (e.g., fit a decreasing trend to data that goes up and down, assert all differences $\le 0$).
