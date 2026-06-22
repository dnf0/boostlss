# Tree Learner Design Spec

- **Status:** Approved
- **Date:** 2026-06-20
- **Scope:** Introduce a Multivariate Decision Tree base learner for the boosting engine.

## 1. Goal
Implement a recursive, multivariate decision tree base learner (`Tree`) that can split on multiple features and handles depths greater than 1. This learner builds on the abstraction layer introduced for `Stump`.

## 2. Architecture

### 2.1 Configuration (`Tree` struct)
The base learner definition requires:
- `feature_names`: `Vec<String>` (Names of the columns the tree can split on).
- `max_depth`: `usize` (Maximum depth of the tree).
- `min_samples_leaf`: `usize` (Minimum number of samples required to form a leaf).

### 2.2 Cached State (`TreeFitState`)
During `initialize(Dataset)`:
- We extract the specified columns.
- To ensure fast split finding at each boosting iteration, we pre-sort the data.
- The state will hold `sorted_features: Vec<Vec<(f64, usize)>>`. This is one sorted vector per feature. Each element is `(value, original_row_idx)`.

### 2.3 The Update Enum (`LearnerUpdate::Tree`)
We will extend `LearnerUpdate` with a recursive tree structure:
```rust
#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf(f64),
    Split {
        feature_idx: usize,
        split_val: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    }
}

pub enum LearnerUpdate {
    Linear(Array1<f64>),
    Stump { split_val: f64, left_val: f64, right_val: f64 },
    Tree(TreeNode),
}
```

### 2.4 Fitting the Tree (`fit_update` Method)
The tree fitting uses a recursive function:
- **Parameters:** Current depth, `active_indices: &[usize]` (or a boolean mask), pseudo-residuals `u`, and weights.
- **Base cases:** Return `Leaf(weighted_mean(u))` if:
  - `depth == max_depth`
  - Number of active indices < `2 * min_samples_leaf`
  - Variance of `u` for active indices is 0 (or very close to 0).
- **Splitting:** Iterate through each feature using its pre-sorted array. Filter for active indices and scan for the split point that maximizes the variance reduction (similar to the Stump logic).
- **Recursing:** Partition the active indices based on the best split and recursively call the function for the left and right children.

### 2.5 Prediction & Engine Integration
- **Prediction:** During `predict` in `model.rs`, row values for the specified features are mapped through the `TreeNode` down to a leaf to retrieve the prediction.
- **Scaling (`nu`):** When committing the update step to the model's history, the engine multiplies the update by `nu` (step length). For `TreeNode`, this means recursively visiting the tree and multiplying every `Leaf` value by `nu`.

### 2.6 Python Bindings
Create `PyTreeLearner` in `boostlss-py`:
```python
model.add_learner("mu", PyTreeLearner(["x1", "x2"], max_depth=2, min_samples_leaf=5))
```
The constructor will accept a list of feature names and parameters. `BoostLssModel.add_learner` will be updated to extract `PyTreeLearner` and convert it to `BaseLearner::Tree`.

## 3. Testing Strategy
- Unit test for `TreeFitState::fit_update` ensuring it splits correctly on toy data.
- Unit test for `TreeNode` scaling by `nu`.
- Python integration test for `PyTreeLearner`.
