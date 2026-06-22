# boostlss — Feature Importance and Partial Dependence Design

## Goal
Provide model interpretability tools by implementing Feature Importance (via Empirical Risk Reduction) and Partial Dependence (via Friedman's method). Expose these features to both the Rust core and the Python bindings.

## 1. Feature Importance (Empirical Risk Reduction)

### Concept
Feature Importance measures how much each base learner contributed to reducing the model's loss (RSS) during training. We calculate this by tracking the RSS before and after the best learner's update at each boosting step, and summing these reductions per learner.

### Implementation Details
*   **Storage (`crates/boostlss/src/model.rs`):**
    *   Update the `UpdateStep` struct to include a `pub risk_reduction: f64` field.
*   **Engine Update (`crates/boostlss/src/engine/cyclical.rs`):**
    *   In `fit_cyclical`, before checking cached learners for a given parameter `k`, calculate the `base_rss` of the current `gradients`:
        ```rust
        let base_rss = match data.weights() {
            Some(w) => (&gradients * &gradients * w).sum(),
            None => (&gradients * &gradients).sum(),
        };
        ```
    *   After identifying the `best_rss` among the candidate learners, compute the risk reduction: `let risk_reduction = base_rss - best_rss;`.
    *   Store this `risk_reduction` in the `UpdateStep` added to `updates`.
*   **API (`crates/boostlss/src/model.rs`):**
    *   Add a method to `Fitted<F>`: `pub fn feature_importance(&self) -> Vec<f64>`.
    *   This method returns a vector of length `self.learners.len()`, where each index corresponds to the total summed `risk_reduction` for that specific base learner.

## 2. Partial Dependence (Friedman's Method)

### Concept
Partial Dependence shows the marginal effect of a feature on the predicted outcome. Using Friedman's method, we replace the target feature's column in the dataset with a constant grid value, compute predictions across all observations, and take the average. This accounts for interactions naturally.

### Implementation Details
*   **API (`crates/boostlss/src/model.rs`):**
    *   Add a method to `Fitted<F>`: `pub fn partial_dependence(&mut self, data: &Dataset, param: &str, feature_idx: usize, grid: &[f64]) -> Result<Vec<f64>, BoostlssError>`.
*   **Logic:**
    *   Iterate over each `value` in `grid`.
    *   For each value, create a cloned copy of `data` where the column at `feature_idx` is replaced entirely by `value`.
    *   Call `self.predict(&modified_data, param, Scale::Link)` and compute the mean of the resulting predictions.
    *   Collect these means into a `Vec<f64>` matching the order of the `grid`.

## 3. Python Bindings

### API (`crates/boostlss-py/src/model.rs`)
*   Expose `feature_importance(&self) -> PyResult<Vec<f64>>` on the `Fitted` Python class.
*   Expose `partial_dependence(&mut self, data: &PyDataset, param: &str, feature_idx: usize, grid: Vec<f64>) -> PyResult<Vec<f64>>` on the `Fitted` Python class.
