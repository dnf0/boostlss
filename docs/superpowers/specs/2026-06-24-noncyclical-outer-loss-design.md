# Non-Cyclical Outer-Loss Design Spec

- Status: Approved
- Date: 2026-06-24

## 1. Overview
The current `NonCyclic` algorithm in `boostlss` implements the **inner-loss** variant. In each boosting iteration, it first shortlists the best base-learner for each distribution parameter by evaluating Residual Sum of Squares (RSS) against the negative gradients. Then, it evaluates only those shortlisted learners against the global Negative Log-Likelihood (NLL) to pick the final update.

This spec details the implementation of the **outer-loss** variant (as originally described by Thomas et al., 2018), where we evaluate *every* candidate base-learner across *all* parameters directly against the global NLL, picking the single `(parameter, base_learner)` pair that minimizes the total NLL in a single stage.

## 2. API Changes

### 2.1 Rust Core
To follow existing patterns while preserving backward compatibility, we will add a new variant to the `Algorithm` enum in `crates/boostlss/src/engine/mod.rs`:

```rust
pub enum Algorithm {
    Cyclic,
    NonCyclic,       // The existing inner-loss variant
    NonCyclicOuter,  // The new outer-loss variant
}
```

### 2.2 Python Bindings
In `crates/boostlss-py/src/model.rs`, the `algorithm` constructor argument will be updated to accept `"noncyclic_outer"`:

```python
# Usage in Python:
model = BoostLssModel(family, mstop=100, step_length=0.1, algorithm="noncyclic_outer")
```

## 3. Engine Implementation

A new function `fit_noncyclical_outer` will be added to `crates/boostlss/src/engine/noncyclical.rs` (or the existing function will be refactored to support both via an argument/match).

**The Outer-Loss Loop:**
For each boosting iteration `m = 1..mstop`:
1. Record the baseline global NLL.
2. Initialize `best_nll = f64::INFINITY`, `best_update = None`.
3. For each parameter `k`:
   - Compute and stabilize negative gradients.
   - For each learner `l` registered for parameter `k`:
     - Fit learner `l` to the gradients, producing a tentative update `u_hat`.
     - Apply `step = u_hat * step_length`.
     - Temporarily update `current_predictions[k] += step`.
     - Calculate `nll = family.nll(data, &current_predictions)`.
     - If `nll < best_nll`: record `best_nll`, `best_update = (k, l, u_hat)`.
     - Revert `current_predictions[k] -= step`.
4. Commit the best update to `current_predictions`, calculate risk reduction, and record the `UpdateStep`.

## 4. Testing Strategy

1. **Rust Engine Tests:**
   Add a unit test in `engine/noncyclical.rs` (e.g., `test_fit_noncyclical_outer_evaluates_all_combinations`) that asserts the correct learner is selected based purely on NLL reduction, without an intermediate RSS shortlisting stage.
2. **PyO3 Integration:**
   Add a test in `crates/boostlss-py/tests/test_model.py` to assert that `algorithm="noncyclic_outer"` compiles, fits successfully, and generates valid predictions.
3. **Property/Validation Tests:**
   Ensure `cvrisk` execution works seamlessly when the model is configured with `NonCyclicOuter`.
