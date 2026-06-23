# Random Effects Base-Learner Design Spec

- Status: Draft for review
- Date: 2026-06-23

## Goal
Implement a Random Effects base-learner (`RandomEffects`) for `boostlss` to support grouped/hierarchical data, matching the behavior of `brandom()` in `mboost`.

## Architecture & Data Model (Option B)

Although `boostlss` v1 natively expects `f64` columns, forcing users to manually one-hot encode factors with hundreds of levels (e.g., user IDs) is unergonomic.

To solve this pragmatically:
1. The user provides a single column containing **0-indexed group indices** encoded as `f64` (e.g., `0.0`, `1.0`, `2.0`, ..., `C-1.0` where `C` is the number of classes).
2. The `RandomEffects` learner reads this column and internally dynamically expands it into an $N \times C$ dummy/one-hot design matrix within its `build_design` method.

## Implementation Details

### 1. Rust Struct Definition
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffects {
    pub feature: String,
    pub df: f64,
}

impl RandomEffects {
    pub fn new(feature: &str) -> Self {
        Self {
            feature: feature.to_string(),
            df: 4.0, // Default df for brandom
        }
    }

    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }
}
```

### 2. Design Matrix (`build_design`)
- During initialization, the learner identifies the maximum index $C-1$ in the training data to determine the number of columns $C$.
- It constructs an $N \times C$ matrix of zeros.
- For each row $i$, if $idx = X[i]$, it sets `design[i, idx] = 1.0`.
- Input validation should ensure values are valid non-negative integers (e.g., `x.fract() == 0.0` and `x >= 0.0`).

### 3. Penalty Matrix (`penalty_matrix`)
- The penalty matrix for random effects (ridge regression) is simply the $C \times C$ identity matrix $I_C$.

### 4. Lambda & Degrees of Freedom
- Uses the existing Demmler-Reinsch routine (`df_to_lambda`) to map the target `df` to the smoothing parameter $\lambda$ using the design matrix and identity penalty matrix.

### 5. Prediction & Unseen Levels
- In standard mixed-effect models, the expected random effect for an unseen group is exactly zero (the global intercept absorbs the mean).
- If the `predict` step encounters a group index $idx \ge C$ or negative, it will safely return `0.0` for that row, emitting a structured warning (similar to P-Spline extrapolation).

## Python Bindings
- We will expose a `PyRandomEffectsLearner` in Python.
- Users can pass categorical pandas columns by taking `.cat.codes` upstream and passing the resulting integer array to the learner.
