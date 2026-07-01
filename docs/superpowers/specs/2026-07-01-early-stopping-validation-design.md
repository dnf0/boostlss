# Design: Early Stopping & Validation Sets

## Overview
This feature introduces early stopping and validation set evaluation to the `boostlss` algorithm. It allows users to monitor the negative log-likelihood (loss) on both a training set and an optional validation set during the boosting iterations. If early stopping is configured, the algorithm will halt training if the validation loss does not improve for a specified number of rounds.

## API Changes

### Rust API
1. **Fit Method Updates:**
   The core `fit` method on `BoostLss` and the underlying engine methods (`fit_cyclical`, `fit_noncyclical`, `fit_noncyclical_outer`) will be updated to accept two new parameters:
   - `eval_data: Option<&Dataset>`: An optional dataset to evaluate validation loss.
   - `early_stopping_rounds: Option<usize>`: The number of rounds without validation loss improvement before halting.

2. **Fitted Model Updates:**
   The `Fitted` struct (returned by `fit`) will be expanded to store evaluation results:
   - `best_iteration: usize`: The iteration (mstop) that achieved the lowest validation loss.
   - `eval_results: EvalResults`: A struct tracking the `train_loss` and `val_loss` at each iteration.

### Python API
1. **`BoostLssModel.fit`:**
   The Python `fit` method will be updated:
   ```python
   def fit(self, X, y, weights=None, eval_set=None, early_stopping_rounds=None): ...
   ```
   - `eval_set`: A tuple `(X_val, y_val)` representing the validation data.
   - `early_stopping_rounds`: An integer specifying the patience for early stopping.

2. **Model Properties:**
   After fitting, the model will expose:
   - `evals_result_`: A dictionary containing the training and validation history. E.g., `{"train": {"loss": [...]}, "valid": {"loss": [...]}}`.
   - `best_iteration_`: An integer indicating the iteration where the best validation loss was found.

## Implementation Details

### Loss Tracking & Early Stopping Logic
At the end of each iteration `m` (after all parameters are updated for cyclic, or after the single best parameter is updated for non-cyclic):
1. **Train Loss:** Calculate the `nll` on the training dataset using the current predictions.
2. **Validation Loss:** If `eval_data` is provided, calculate predictions for the validation set using the current model state, and compute the `nll`.
3. **Best Iteration:** Maintain a `best_val_nll` (initialized to infinity) and a `best_iteration` variable. If the current validation `nll` < `best_val_nll`, update both.
4. **Early Stopping Trigger:** If `early_stopping_rounds` is provided and `m - best_iteration >= early_stopping_rounds`, break the training loop.
5. **Truncation:** If early stopping is triggered, the `updates` vector inside the `Fitted` model must be truncated to `best_iteration` to restore the best weights and prevent overfitting.

### Prediction & Validation State
To efficiently compute validation loss, the engines will maintain a running vector of `current_val_predictions`. When an update step is applied to the training set, the corresponding base learner prediction is also computed for the validation set and added to `current_val_predictions`. This avoids doing a full `.predict()` from scratch at each iteration.

## Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResults {
    pub train_loss: Vec<f64>,
    pub val_loss: Option<Vec<f64>>,
}
```

Modifications to `Fitted<F>`:
```rust
pub struct Fitted<F: Family> {
    // ... existing fields ...
    pub eval_results: EvalResults,
    pub best_iteration: usize,
}
```

## Testing
- **Rust Unit Tests:** Add tests to verify that `fit_cyclical` and `fit_noncyclical` halt correctly when early stopping is configured and that `updates` are truncated appropriately.
- **Python Integration Tests:** Add tests in `boostlss-py/tests` to verify that passing `eval_set` and `early_stopping_rounds` works, and that `evals_result_` and `best_iteration_` properties are populated.
