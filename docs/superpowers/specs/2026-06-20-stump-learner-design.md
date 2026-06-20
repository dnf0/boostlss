# Stump Learner and Abstraction Refactor Design

## Context
`boostlss` currently hardcodes the engine loop to assume that all base learners are linear models (or linear combinations of basis functions like P-Splines). The engine expects to extract a static design matrix ($X$) and a penalty matrix ($K$) before the boosting loop starts, computes a Cholesky factorization of $(X^T X + \lambda K)$, and caches it.

To support decision stumps (depth-1 trees) and eventually deeper trees, we need to decouple the engine from the Cholesky solver. Trees do not use static design matrices; they find optimal splits dynamically based on the pseudo-residuals at each iteration.

## Architecture

### 1. Abstracting the Learner Interface
We will reverse the dependency: the boosting engine will no longer manage Cholesky factorizations. Instead, learners will manage their own fitting logic.

*   `BaseLearner` will expose an initialization method:
    `fn initialize(&self, x: &Array1<f64>) -> Result<LearnerFit, BoostlssError>`
*   `LearnerFit` will become an enum encapsulating the internal state required for fast updates:
    ```rust
    pub enum LearnerFit {
        Linear(LinearFitState),
        PSpline(PSplineFitState),
        Stump(StumpFitState),
    }
    ```
    *Note: `LinearFitState` and `PSplineFitState` will hold the cached Cholesky `Llt` objects.*
*   `LearnerFit` will expose the core update method:
    `fn fit_update(&self, u: ArrayView1<f64>, weights: Option<ArrayView1<f64>>) -> LearnerUpdate`

### 2. Generalizing Model Updates and Predictions
The `UpdateStep` currently stores `coef: Array1<f64>`, which is inherently tied to linear learners. We will change this to use a `LearnerUpdate` enum.

```rust
pub enum LearnerUpdate {
    Linear(Array1<f64>), // Stores coefficients
    Stump { split_val: f64, left_val: f64, right_val: f64 },
}
```

*   The `Fitted` model's `predict` method will iterate over updates.
    *   For `LearnerUpdate::Linear`, it will perform standard matrix multiplication ($X \beta$). (This requires the model to reconstruct the design matrix during prediction, which it currently does).
    *   For `LearnerUpdate::Stump`, it will threshold the input vector: `if x_i <= split_val { left_val } else { right_val }`.

### 3. The Stump Learner Implementation
*   **Initialization (`StumpFitState`)**: The learner receives the feature vector $x$. It will create an argsort index or simply a sorted copy of $x$ along with the original indices. This pre-sorting allows for an $O(N)$ split search during each boosting iteration.
*   **Fitting**: During `fit_update`, given pseudo-residuals $u$ and optional weights $w$:
    *   It iterates through the sorted $x$ values, treating each unique midpoint as a potential split.
    *   It maintains running sums of $w$, $w \cdot u$, and $w \cdot u^2$ for the left and right child nodes.
    *   It calculates the reduction in squared error (or variance) for each split.
    *   It selects the split point that minimizes the weighted squared error and computes the optimal constant values (`left_val`, `right_val`) for the leaves.

### 4. Python Bindings
We will expose the stump learner to Python so it can be used seamlessly alongside existing learners.
*   Create `PyStumpLearner` in `crates/boostlss-py/src/learner.rs`.
*   Users will invoke it via: `model.add_learner("mu", PyStumpLearner("feature_name"))`.

## Error Handling
*   If a Stump fails to find a valid split (e.g., all $x$ values are identical), it should return a split that predicts the global mean of $u$ for both left and right, effectively becoming an intercept-only model for that iteration.

## Testing
*   Unit tests for `StumpFitState::fit_update` verifying correct split points on synthetic data.
*   Integration tests verifying that a `BoostLss` model containing both `Linear` and `Stump` learners can fit and predict without panicking.
*   Python tests verifying `PyStumpLearner` can be added and trained.
