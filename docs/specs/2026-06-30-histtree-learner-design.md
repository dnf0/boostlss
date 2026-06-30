# HistTree Learner Design Specification

## Overview

Implement a `HistTree` base learner to provide high-speed, histogram-based approximate greedy tree building. This allows users to choose between `Tree` (exact greedy, closer to `mboost`) and `HistTree` (histogram approximation, closer to `lightgbm` / `xgboost` performance profiles).

## API & Architecture (Separate Learner)

- **Rust API:** A new `HistTree` struct implementing `BaseLearner` (similar to `Tree`), with a configurable `max_bins` parameter (default 256).
- **Python API:** `boostlss_py.HistTree(features=[...], max_depth=3, min_samples_leaf=1, max_bins=256)`.
- **Enum Integration:** Add `BaseLearner::HistTree` and `LearnerFit::HistTree` to `crates/boostlss/src/learner/mod.rs`.

## Quantization Strategy (Initialization)

During `HistTree::initialize()` (i.e. `build_fit_state`):

1. For each feature, compute exact uniform quantiles by sorting the feature column and taking `(N / max_bins)` stepped percentiles.
2. If a feature has fewer unique values than `max_bins`, use the exact unique values as boundaries.
3. Transform the `f64` continuous dataset into a `Array2<u8>` quantized matrix.
4. Store the original `f64` boundaries alongside the quantized matrix in `HistTreeFitState`.

## Histogram Construction (Tree Building)

During `fit_update()` (recursive split finding):

1. **Histogram Array:** For a given node and feature, instantiate an array `[(f64, f64); max_bins]` storing the sum of gradients (`u * w`) and sum of weights (`w`).
2. **Population:** Iterate over the `active_indices` in the `u8` matrix for that feature. Use the `u8` value directly as an index to add the row's gradient and weight to the histogram bin. This is strictly $O(N)$ and extremely cache-efficient.
3. **Split Search:** Iterate linearly from `bin = 0` to `max_bins - 1`. Accumulate `left_w` and `left_wu`, subtract from `total_w` and `total_wu` to get `right_w` and `right_wu`. Calculate the gain `(left_wu^2 / left_w) + (right_wu^2 / right_w)`. Track the bin that provides the maximum gain.
4. **Resolution:** Translate the best `u8` split bin back into its corresponding `f64` threshold using the boundaries stored during initialization.
5. **Subtraction Trick:** (Optional optimization) Calculate histograms only for the smaller child node and subtract from the parent's histogram to get the larger child's histogram.

## Integration Points

- `crates/boostlss/src/learner/hist_tree.rs` (New file)
- `crates/boostlss/src/learner/mod.rs` (Enum wiring)
- `crates/boostlss-py/src/learner.rs` (Python wrapper)
