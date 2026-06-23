# Non-Cyclical (`NonCyclic`) Boosting Algorithm Design

## Context

The `boostlss` engine currently supports `Cyclic` boosting, where each parameter is updated in a round-robin fashion during every boosting iteration. The goal is to implement `NonCyclic` boosting, where only the single best parameter-learner update is selected and applied per iteration, minimizing the total Negative Log-Likelihood (NLL) of the model.

## Two-Stage Selection Architecture

As proposed and approved, we will use a **Two-Stage Selection** approach within the `m = 1..mstop` loop. This avoids evaluating the expensive NLL function $K \times J$ times, evaluating it only $K$ times instead.

### Stage 1: Per-Parameter Candidate Selection (RSS)

1. For each parameter $k \in K$:
   - Calculate the pseudo-residuals (negative gradient) for parameter $k$.
   - Fit all registered base-learners for parameter $k$ using Residual Sum of Squares (RSS) against the pseudo-residuals.
   - Select the single best base-learner $j$ for parameter $k$.
   - Compute the corresponding parameter update (scaled by `nu` / step length).
2. We now have exactly $K$ candidate updates (one optimal update for each parameter).

### Stage 2: Cross-Parameter Candidate Selection (NLL)

1. Evaluate the base NLL of the current predictions (before any updates are applied).
2. For each candidate update $k$:
   - Temporarily apply the candidate update to the model's predictions.
   - Calculate the NLL of the model using `family.nll(data, current_predictions)`.
   - Revert the candidate update.
3. Select the candidate update $k^*$ that yields the lowest NLL overall.
4. If the lowest NLL is not better than the base NLL, we can either continue (as boosting sometimes allows small upticks in NLL due to step length constraints) or we simply apply it since it was the best among the choices. In standard gradient boosting, the best candidate is applied regardless of an absolute NLL decrease (because gradients dictate the direction). But selecting the _minimum_ NLL among the candidates ensures we step in the most optimal direction.

### Mstop Handling

In `Cyclic` boosting, `Mstop::PerParam(Vec<usize>)` assigns an iteration budget to each parameter.
For `NonCyclic` boosting, `Mstop` indicates the total number of algorithm iterations (i.e. number of updates chosen).

- If `Mstop::Scalar(m)` is provided, we run exactly `m` iterations.
- If `Mstop::PerParam(v)` is provided, it is incompatible with the philosophical concept of `NonCyclic` because `NonCyclic` dynamically allocates updates among parameters based on NLL improvement. We will return a `BoostlssError::InvalidConfig` if `Mstop::PerParam` is used with `Algorithm::NonCyclic`.

## Implementation Details

### File Additions

- **`crates/boostlss/src/engine/noncyclical.rs`**: Create this file to hold the `fit_noncyclical` function.

### File Modifications

- **`crates/boostlss/src/engine/mod.rs`**: Expose `pub mod noncyclical;`.
- **`crates/boostlss/src/model.rs`**: Update `BoostLss::fit` to route `Algorithm::NonCyclic` to `fit_noncyclical` instead of throwing a "Not yet implemented" error.

### Error Handling

- Validate that `config.mstop` is `Mstop::Scalar`. If not, return `BoostlssError::InvalidConfig("NonCyclic algorithm requires a Scalar Mstop".into())`.
- Bubble up any errors from `family.ngradient`, `family.nll`, or base-learner initialization.

### Testing

- Write `test_fit_noncyclical_selects_best_param` in `noncyclical.rs` using a synthetic dataset where one parameter clearly needs updating more than another, and assert that the non-cyclical engine correctly prioritizes the updates for that parameter.
- Ensure the predictions match expected monotonic trends.
