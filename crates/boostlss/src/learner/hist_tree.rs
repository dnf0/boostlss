use crate::data::Dataset;
use crate::learner::LearnerUpdate;
use ndarray::{Array2, ArrayView1, ShapeBuilder};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistTree {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_bins: usize,
    pub categorical_features: Vec<usize>,
}

impl HistTree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            feature_indices,
            max_depth: 3,
            min_samples_leaf: 1,
            max_bins: 256,
            categorical_features: Vec::new(),
        }
    }

    pub fn categorical_features(mut self, cat_features: Vec<usize>) -> Self {
        self.categorical_features = cat_features;
        self
    }

    pub fn max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn build_fit_state(
        &self,
        data: &Dataset,
    ) -> Result<HistTreeFitState, crate::error::BoostlssError> {
        let n_obs = data.n_obs();
        let mut quantized_data = Array2::<u8>::zeros((n_obs, self.feature_indices.len()).f());
        let mut thresholds = Vec::with_capacity(self.feature_indices.len());

        for (feat_idx_out, &feat_idx_in) in self.feature_indices.iter().enumerate() {
            let col = data.design().get_column(feat_idx_in).unwrap();

            // 1. Get exact unique sorted values
            let mut unique_vals: Vec<f64> = col.iter().copied().collect();
            unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_vals.dedup();

            // 2. Sample to get quantiles if there are too many unique values
            let boundaries = if self.categorical_features.contains(&feat_idx_in) {
                let mut b = unique_vals;
                if b.len() > self.max_bins {
                    b.truncate(self.max_bins);
                }
                b
            } else if unique_vals.len() <= self.max_bins {
                unique_vals
            } else {
                let step = unique_vals.len() as f64 / self.max_bins as f64;
                let mut sample: Vec<f64> = (0..self.max_bins)
                    .map(|i| {
                        let idx = (i as f64 * step) as usize;
                        unique_vals[idx.min(unique_vals.len() - 1)]
                    })
                    .collect();
                sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sample.dedup();
                sample
            };

            thresholds.push(boundaries.clone());

            // 3. Map continuous data to bins (0 to len-1)
            for (i, &val) in col.iter().enumerate() {
                // Find the first boundary that is > val.
                // Since `partition_point` returns the index of the first element not satisfying the predicate (i.e. >= val),
                // we actually want the index of the interval it falls into.
                // A simpler binary search:
                let bin = boundaries
                    .binary_search_by(|v| v.partial_cmp(&val).unwrap())
                    .unwrap_or_else(|e| e);
                // Clamp bin to max u8
                quantized_data[[i, feat_idx_out]] = bin.min(255) as u8;
            }
        }

        Ok(HistTreeFitState {
            feature_indices: self.feature_indices.clone(),
            quantized_data,
            thresholds,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            max_bins: self.max_bins,
            categorical_features: self.categorical_features.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistTreeFitState {
    pub feature_indices: Vec<usize>,
    pub quantized_data: Array2<u8>,
    pub thresholds: Vec<Vec<f64>>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_bins: usize,
    pub categorical_features: Vec<usize>,
}

impl HistTreeFitState {
    pub fn fit_update(
        &self,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> LearnerUpdate {
        let active_indices: Vec<usize> = (0..u.len())
            .filter(|&i| {
                if let Some(w) = weights {
                    w[i] > 0.0
                } else {
                    true
                }
            })
            .collect();

        let root = self.build_tree(active_indices, 0, u, weights);
        LearnerUpdate::HistTree {
            node: root,
            feature_idx: self.feature_indices[0],
        }
    }

    fn build_tree(
        &self,
        active_indices: Vec<usize>,
        depth: usize,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> crate::learner::tree::TreeNode {
        let u_slice = u.as_slice().unwrap();
        let w_slice = weights.as_ref().map(|w| w.as_slice().unwrap());

        let mut total_w = 0.0;
        let mut total_wu = 0.0;
        if let Some(w) = w_slice {
            for &idx in &active_indices {
                total_w += w[idx];
                total_wu += w[idx] * u_slice[idx];
            }
        } else {
            for &idx in &active_indices {
                total_w += 1.0;
                total_wu += u_slice[idx];
            }
        }

        if depth >= self.max_depth
            || active_indices.len() < self.min_samples_leaf * 2
            || total_w <= 1e-9
        {
            let mean = if total_w > 1e-9 {
                total_wu / total_w
            } else {
                0.0
            };
            return crate::learner::tree::TreeNode::Leaf {
                value: mean,
                samples: active_indices.len(),
            };
        }

        let mut best_gain = if total_w > 1e-9 {
            (total_wu * total_wu) / total_w
        } else {
            -1.0
        };
        enum SplitCandidate {
            Continuous(u8),
            Categorical(Vec<u8>),
        }
        let mut best_split: Option<(usize, SplitCandidate)> = None;

        let max_possible_bins = self.max_bins + 1;
        let mut hist_w = vec![0.0; max_possible_bins];
        let mut hist_wu = vec![0.0; max_possible_bins];
        let mut hist_count = vec![0; max_possible_bins];

        for (feat_out, _) in self.feature_indices.iter().enumerate() {
            let max_bin = self.thresholds[feat_out].len();
            let is_categorical = self
                .categorical_features
                .contains(&self.feature_indices[feat_out]);

            // Fast reset
            hist_w[..max_bin].fill(0.0);
            hist_wu[..max_bin].fill(0.0);
            hist_count[..max_bin].fill(0);

            let col = self.quantized_data.column(feat_out);
            let col_slice = col.as_slice().unwrap();

            if let Some(w) = w_slice {
                for &idx in &active_indices {
                    let bin = col_slice[idx] as usize;
                    hist_w[bin] += w[idx];
                    hist_wu[bin] += w[idx] * u_slice[idx];
                    hist_count[bin] += 1;
                }
            } else {
                for &idx in &active_indices {
                    let bin = col_slice[idx] as usize;
                    hist_w[bin] += 1.0;
                    hist_wu[bin] += u_slice[idx];
                    hist_count[bin] += 1;
                }
            }

            // 2. Scan Histogram for split
            if is_categorical {
                let mut cat_bins = Vec::new();
                for bin in 0..max_bin {
                    if hist_count[bin] > 0 {
                        cat_bins.push((bin as u8, hist_w[bin], hist_wu[bin], hist_count[bin]));
                    }
                }

                cat_bins.sort_by(|a, b| {
                    let mean_a = if a.1 > 1e-9 { a.2 / a.1 } else { 0.0 };
                    let mean_b = if b.1 > 1e-9 { b.2 / b.1 } else { 0.0 };
                    mean_a
                        .partial_cmp(&mean_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut left_w = 0.0;
                let mut left_wu = 0.0;
                let mut left_count = 0;

                for i in 0..cat_bins.len().saturating_sub(1) {
                    left_w += cat_bins[i].1;
                    left_wu += cat_bins[i].2;
                    left_count += cat_bins[i].3;

                    let right_count = active_indices.len() - left_count;
                    if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                        continue;
                    }

                    let right_w = total_w - left_w;
                    let right_wu = total_wu - left_wu;

                    if left_w <= 1e-9 || right_w <= 1e-9 {
                        continue;
                    }

                    let gain = (left_wu * left_wu / left_w) + (right_wu * right_wu / right_w);
                    if gain > best_gain {
                        best_gain = gain;
                        let left_bins: Vec<u8> = cat_bins[0..=i].iter().map(|c| c.0).collect();
                        best_split = Some((feat_out, SplitCandidate::Categorical(left_bins)));
                    }
                }
            } else {
                let mut left_w = 0.0;
                let mut left_wu = 0.0;

                for bin in 0..max_bin {
                    left_w += hist_w[bin];
                    left_wu += hist_wu[bin];

                    let right_w = total_w - left_w;
                    let right_wu = total_wu - left_wu;

                    if left_w <= 1e-9 || right_w <= 1e-9 {
                        continue;
                    }

                    let gain = (left_wu * left_wu / left_w) + (right_wu * right_wu / right_w);

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some((feat_out, SplitCandidate::Continuous(bin as u8)));
                    }
                }
            }
        }

        if let Some((feat_out, split_val)) = best_split {
            // Reconstruct the indices only for the chosen best split
            let col = self.quantized_data.column(feat_out);
            let col_slice = col.as_slice().unwrap();

            let mut left_idx = Vec::with_capacity(active_indices.len() / 2);
            let mut right_idx = Vec::with_capacity(active_indices.len() / 2);

            match split_val {
                SplitCandidate::Continuous(best_bin) => {
                    for &idx in &active_indices {
                        if col_slice[idx] <= best_bin {
                            left_idx.push(idx);
                        } else {
                            right_idx.push(idx);
                        }
                    }

                    if left_idx.len() >= self.min_samples_leaf
                        && right_idx.len() >= self.min_samples_leaf
                    {
                        let threshold = self.thresholds[feat_out][best_bin as usize];

                        let left = Box::new(self.build_tree(left_idx, depth + 1, u, weights));
                        let right = Box::new(self.build_tree(right_idx, depth + 1, u, weights));

                        return crate::learner::tree::TreeNode::Split {
                            feature_idx: feat_out,
                            threshold,
                            left,
                            right,
                        };
                    }
                }
                SplitCandidate::Categorical(left_bins) => {
                    for &idx in &active_indices {
                        if left_bins.contains(&col_slice[idx]) {
                            left_idx.push(idx);
                        } else {
                            right_idx.push(idx);
                        }
                    }

                    if left_idx.len() >= self.min_samples_leaf
                        && right_idx.len() >= self.min_samples_leaf
                    {
                        let left_categories: Vec<f64> = left_bins
                            .iter()
                            .map(|&b| self.thresholds[feat_out][b as usize])
                            .collect();

                        let left = Box::new(self.build_tree(left_idx, depth + 1, u, weights));
                        let right = Box::new(self.build_tree(right_idx, depth + 1, u, weights));

                        return crate::learner::tree::TreeNode::CategoricalSplit {
                            feature_idx: feat_out,
                            left_categories,
                            left,
                            right,
                        };
                    }
                }
            }
        }

        let mean = if total_w > 1e-9 {
            total_wu / total_w
        } else {
            0.0
        };
        crate::learner::tree::TreeNode::Leaf {
            value: mean,
            samples: active_indices.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use ndarray::array;

    #[test]
    fn test_hist_tree_initialization() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Dataset::new(x, y.clone(), None, None).unwrap();

        let hist_tree = HistTree::new(vec![0]).max_bins(3);
        let state = hist_tree.build_fit_state(&data).unwrap();

        // Assert we have quantized data and thresholds
        assert_eq!(state.quantized_data.shape(), &[5, 1]);
        assert_eq!(state.thresholds.len(), 1);
        assert!(state.thresholds[0].len() <= 3);
    }

    #[test]
    fn test_hist_tree_fit() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = crate::data::Dataset::new(x, y.clone(), None, None).unwrap();

        let hist_tree = HistTree::new(vec![0]).max_bins(5).max_depth(1);
        let state = hist_tree.build_fit_state(&data).unwrap();

        let u = array![-1.0, -1.0, 0.0, 1.0, 1.0]; // pseudo-residuals
        let weights = ndarray::Array1::ones(5);

        let update = state.fit_update(u.view(), Some(weights.view()));

        // Assert we built a tree update
        match update {
            crate::learner::LearnerUpdate::HistTree { node, .. } => {
                if let crate::learner::tree::TreeNode::Split { left, right: _, .. } = node {
                    // It should have successfully split
                    assert!(
                        matches!(*left, crate::learner::tree::TreeNode::Leaf { .. })
                            || matches!(*left, crate::learner::tree::TreeNode::Split { .. })
                    );
                } else {
                    panic!("Expected root to be a Split node");
                }
            }
            _ => panic!("Wrong update type"),
        }
    }

    #[test]
    fn test_hist_tree_categorical_split() {
        let x = array![[1.0], [2.0], [3.0], [1.0], [2.0]];
        let y = array![10.0, 20.0, 30.0, 10.0, 20.0];
        let data = crate::data::Dataset::new(x, y.clone(), None, None).unwrap();

        let hist_tree = HistTree::new(vec![0])
            .categorical_features(vec![0])
            .max_bins(5)
            .max_depth(1);
        let state = hist_tree.build_fit_state(&data).unwrap();

        // 1.0 and 3.0 have positive residuals, 2.0 has negative
        let u = array![10.0, -10.0, 10.0, 10.0, -10.0];
        let weights = ndarray::Array1::ones(5);

        let update = state.fit_update(u.view(), Some(weights.view()));

        match update {
            crate::learner::LearnerUpdate::HistTree { node, .. } => {
                if let crate::learner::tree::TreeNode::CategoricalSplit {
                    feature_idx,
                    left_categories,
                    ..
                } = node
                {
                    assert_eq!(feature_idx, 0);
                    let has_1 = left_categories.contains(&1.0);
                    let has_2 = left_categories.contains(&2.0);
                    let has_3 = left_categories.contains(&3.0);

                    if has_1 {
                        assert!(has_3);
                        assert!(!has_2);
                    } else {
                        assert!(has_2);
                        assert!(!has_1);
                        assert!(!has_3);
                    }
                } else {
                    panic!("Expected root to be a CategoricalSplit node");
                }
            }
            _ => panic!("Wrong update type"),
        }
    }
}
