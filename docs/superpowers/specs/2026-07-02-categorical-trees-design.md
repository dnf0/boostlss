# Design: Categorical Feature Support in Trees

## Overview
This feature introduces native support for categorical features in the `Tree` and `HistTree` learners. Rather than relying on standard continuous threshold splits (e.g., `feature <= 2.5`), categorical features will use subset partitioning splits (e.g., `feature in [0, 2, 4]`). The implementation is modeled after LightGBM's approach, where categorical values are sorted dynamically at each node based on their gradient statistics to find optimal subsets efficiently.

## API Changes

### Python API
Users will be expected to ordinally encode their categorical variables upstream (e.g., into integers represented as `f64`). They can then specify which columns are categorical when creating a learner.

```python
class PyTreeLearner:
    def __init__(self, feature_indices, max_depth=3, min_samples_leaf=1, categorical_features=None): ...

class PyHistTreeLearner:
    def __init__(self, feature_indices, max_bins=256, max_depth=3, min_samples_leaf=1, categorical_features=None): ...
```
- `categorical_features`: An optional list of integers specifying the feature indices that should be treated as categorical. Default is an empty list.

### Rust API
The Rust constructors and structs for `Tree` and `HistTree` will be updated to store `categorical_features`:
```rust
pub struct Tree {
    // ... existing fields
    pub categorical_features: Vec<usize>,
}
// Same for HistTree
```

## Architecture

### Node Representation
The core `TreeNode` enum in `crates/boostlss/src/learner/tree.rs` (which serves both `Tree` and `HistTree` predictions) will be expanded to support categorical partitioning:

```rust
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    CategoricalSplit {
        feature_idx: usize,
        left_categories: Vec<f64>,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}
```

### Prediction Logic
During inference (`predict` in `Tree`), if the traversal encounters a `CategoricalSplit`, it checks if the instance's feature value is contained within `left_categories`. If `left_categories.contains(&val)`, it traverses to the `left` child; otherwise, it traverses to the `right` child.

## Splitting Algorithm (LightGBM Style)

When building the tree, if the current feature being evaluated is in the `categorical_features` list, the standard sequential threshold search is replaced with a gradient-based subset search:

1. **Category Aggregation**: Iterate through the active instances and aggregate the sum of weights (`w`), sum of weighted gradients (`w * u`), and sample counts for each unique category present in the current node.
2. **Category Sorting**: Calculate the mean gradient statistic (`sum_wu / sum_w`) for each category. Sort the categories based on this mean statistic.
3. **Linear Scan for Subsets**: Iterate through the sorted categories. Keep running sums of `left_w`, `left_wu` and calculate the corresponding `right_w`, `right_wu`. Compute the gain at each possible split point along the sorted categories.
4. **Node Creation**: Identify the split point that maximizes gain. All categories falling to the left of this split point in the sorted array are collected into a `left_categories` vector. A `CategoricalSplit` node is created using this vector.

### Integration in `Tree` (Exact)
In `TreeFitState::build_tree`, for categorical features, the pre-sorted features array will be scanned to aggregate values per category. Since the data is exact `f64`, we group identical values to form the category aggregates, sort them, evaluate splits, and apply the mask partitioning logic using the resulting `left_categories`.

### Integration in `HistTree` (Histogram)
In `HistTreeFitState::build_fit_state`, quantization will be adjusted. For features marked as categorical, standard quantile binning will be skipped. Instead, the exact values will be mapped directly to bins (up to `max_bins`). During `build_tree`, the histogram building remains similar, but the bins for categorical features are extracted, sorted by their mean gradient, and scanned to evaluate optimal subsets. The resulting `left_categories` will translate the selected bins back to the original `f64` values for inference.

## Testing
- Unit tests verifying the accuracy of the gradient sorting logic for subset splits.
- Integration tests ensuring that specifying `categorical_features` routes data correctly and produces valid trees without panicking.
