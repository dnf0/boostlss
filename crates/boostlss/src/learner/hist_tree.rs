use crate::data::Dataset;
use crate::learner::LearnerUpdate;
use ndarray::{Array2, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistTree {
    pub feature_indices: Vec<usize>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_bins: usize,
}

impl HistTree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            feature_indices,
            max_depth: 3,
            min_samples_leaf: 1,
            max_bins: 256,
        }
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
        let mut quantized_data = Array2::<u8>::zeros((n_obs, self.feature_indices.len()));
        let mut thresholds = Vec::with_capacity(self.feature_indices.len());

        for (feat_idx_out, &feat_idx_in) in self.feature_indices.iter().enumerate() {
            let col = data.design().get_column(feat_idx_in).unwrap();

            // 1. Get exact unique sorted values
            let mut unique_vals: Vec<f64> = col.iter().copied().collect();
            unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_vals.dedup();

            // 2. Sample to get quantiles if there are too many unique values
            let boundaries = if unique_vals.len() <= self.max_bins {
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
        let mut total_w = 0.0;
        let mut total_wu = 0.0;
        for &idx in &active_indices {
            let w = weights.map(|w| w[idx]).unwrap_or(1.0);
            total_w += w;
            total_wu += w * u[idx];
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
        let mut best_split: Option<(usize, u8, f64, f64, Vec<usize>, Vec<usize>)> = None;

        for (feat_out, _) in self.feature_indices.iter().enumerate() {
            let max_bin = self.thresholds[feat_out].len();
            // 1. Build Histogram
            let mut hist_w = vec![0.0; max_bin + 1];
            let mut hist_wu = vec![0.0; max_bin + 1];

            for &idx in &active_indices {
                let bin = self.quantized_data[[idx, feat_out]] as usize;
                let w = weights.map(|w| w[idx]).unwrap_or(1.0);
                hist_w[bin] += w;
                hist_wu[bin] += w * u[idx];
            }

            // 2. Scan Histogram for split
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

                    // Split the indices explicitly for the children
                    let mut left_indices = Vec::with_capacity(active_indices.len() / 2);
                    let mut right_indices = Vec::with_capacity(active_indices.len() / 2);

                    for &idx in &active_indices {
                        if self.quantized_data[[idx, feat_out]] <= bin as u8 {
                            left_indices.push(idx);
                        } else {
                            right_indices.push(idx);
                        }
                    }

                    best_split = Some((
                        feat_out,
                        bin as u8,
                        left_w,
                        right_w,
                        left_indices,
                        right_indices,
                    ));
                }
            }
        }

        if let Some((feat_out, best_bin, _lw, _rw, left_idx, right_idx)) = best_split {
            if left_idx.len() >= self.min_samples_leaf && right_idx.len() >= self.min_samples_leaf {
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
        let data = Dataset::new(x, y.clone(), None).unwrap();

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
        let data = crate::data::Dataset::new(x, y.clone(), None).unwrap();

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
}
