use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
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
}

impl TreeNode {
    pub fn scale(&mut self, factor: f64) {
        match self {
            TreeNode::Leaf { value, .. } => {
                *value *= factor;
            }
            TreeNode::Split { left, right, .. } => {
                left.scale(factor);
                right.scale(factor);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tree {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub feature_indices: Vec<usize>,
}

impl Tree {
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            max_depth: 3,
            min_samples_leaf: 1,
            feature_indices,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    pub fn build_fit_state(
        &self,
        data: &crate::data::Dataset,
    ) -> Result<TreeFitState, crate::error::BoostlssError> {
        // Inside tree.rs build_design or fit
        let n_cols = data.n_features();
        let mut dense_mat = ndarray::Array2::zeros((data.n_obs(), n_cols));
        for i in 0..n_cols {
            let col = data.design().get_column(i)?;
            dense_mat.column_mut(i).assign(&col);
        }

        let mut sorted_features = Vec::with_capacity(self.feature_indices.len());
        for &col_idx in &self.feature_indices {
            let col = dense_mat.column(col_idx);
            let mut sorted: Vec<(f64, usize)> = col
                .iter()
                .copied()
                .enumerate()
                .map(|(i, val)| (val, i))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            sorted_features.push(sorted);
        }
        Ok(TreeFitState {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            feature_indices: self.feature_indices.clone(),
            sorted_features,
        })
    }

    pub fn predict(
        &self,
        root: &TreeNode,
        data: &crate::data::Dataset,
    ) -> Result<ndarray::Array1<f64>, crate::error::BoostlssError> {
        let n_cols = data.n_features();
        let mut dense_mat = ndarray::Array2::zeros((data.n_obs(), n_cols));
        for i in 0..n_cols {
            let col = data.design().get_column(i)?;
            dense_mat.column_mut(i).assign(&col);
        }

        let mut u_hat = ndarray::Array1::zeros(data.n_obs());
        for i in 0..u_hat.len() {
            let mut node_ptr = root;
            loop {
                match node_ptr {
                    TreeNode::Leaf { value, .. } => {
                        u_hat[i] = *value;
                        break;
                    }
                    TreeNode::Split {
                        feature_idx,
                        threshold,
                        left,
                        right,
                    } => {
                        let col_idx = self.feature_indices[*feature_idx];
                        let val = dense_mat.column(col_idx)[i];
                        if val <= *threshold {
                            node_ptr = left;
                        } else {
                            node_ptr = right;
                        }
                    }
                }
            }
        }
        Ok(u_hat)
    }
}

use crate::learner::LearnerUpdate;
use ndarray::ArrayView1;

#[derive(Debug, Clone)]
pub struct TreeFitState {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub feature_indices: Vec<usize>,
    // Each outer vec corresponds to one feature.
    // Inner vec is the sorted (value, original_row_index) pairs.
    pub sorted_features: Vec<Vec<(f64, usize)>>,
}

impl TreeFitState {
    pub fn fit_update(
        &self,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> LearnerUpdate {
        let active_indices: Vec<usize> = (0..u.len()).collect();
        let w = weights
            .map(|w| w.to_owned())
            .unwrap_or_else(|| ndarray::Array1::ones(u.len()));
        LearnerUpdate::Tree {
            node: self.build_tree(0, &active_indices, u, w.view()),
            param: String::new(),
        }
    }

    fn build_tree(
        &self,
        depth: usize,
        active_indices: &[usize],
        u: ArrayView1<f64>,
        w: ArrayView1<f64>,
    ) -> TreeNode {
        let mut total_w = 0.0;
        let mut total_wu = 0.0;
        for &idx in active_indices {
            total_w += w[idx];
            total_wu += w[idx] * u[idx];
        }

        if depth >= self.max_depth
            || active_indices.len() < 2 * self.min_samples_leaf
            || total_w <= 1e-9
        {
            let mean = if total_w > 1e-9 {
                total_wu / total_w
            } else {
                0.0
            };
            return TreeNode::Leaf {
                value: mean,
                samples: active_indices.len(),
            };
        }

        let mut best_gain = if total_w > 1e-9 {
            (total_wu * total_wu) / total_w
        } else {
            -1.0
        };
        let mut best_split = None;

        for (feat_idx, sorted) in self.sorted_features.iter().enumerate() {
            // Filter sorted elements to only active ones
            let active_sorted: Vec<&(f64, usize)> = sorted
                .iter()
                .filter(|(_, orig_idx)| active_indices.contains(orig_idx))
                .collect();

            let mut left_w = 0.0;
            let mut left_wu = 0.0;
            let mut left_count = 0;

            for i in 0..active_sorted.len().saturating_sub(1) {
                let (val, orig_idx) = active_sorted[i];
                let weight = w[*orig_idx];
                left_w += weight;
                left_wu += weight * u[*orig_idx];
                left_count += 1;

                let right_count = active_sorted.len() - left_count;
                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    continue;
                }

                let next_val = active_sorted[i + 1].0;
                if (val - next_val).abs() < 1e-9 {
                    continue;
                }

                let right_w = total_w - left_w;
                let right_wu = total_wu - left_wu;

                if left_w > 1e-9 && right_w > 1e-9 {
                    let gain = (left_wu * left_wu) / left_w + (right_wu * right_wu) / right_w;
                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some((feat_idx, (val + next_val) / 2.0));
                    }
                }
            }
        }

        if let Some((feat_idx, split_val)) = best_split {
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();
            for &idx in active_indices {
                // Find value for this row
                let val = self.sorted_features[feat_idx]
                    .iter()
                    .find(|(_, i)| *i == idx)
                    .unwrap()
                    .0;
                if val <= split_val {
                    left_indices.push(idx);
                } else {
                    right_indices.push(idx);
                }
            }
            let left = Box::new(self.build_tree(depth + 1, &left_indices, u, w));
            let right = Box::new(self.build_tree(depth + 1, &right_indices, u, w));
            TreeNode::Split {
                feature_idx: feat_idx,
                threshold: split_val,
                left,
                right,
            }
        } else {
            let mean = if total_w > 1e-9 {
                total_wu / total_w
            } else {
                0.0
            };
            TreeNode::Leaf {
                value: mean,
                samples: active_indices.len(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tree_fit_update() {
        let sorted_f0 = vec![(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3)];
        let sorted_f1 = vec![(0.5, 3), (1.5, 2), (2.5, 1), (3.5, 0)];
        let state = TreeFitState {
            max_depth: 2,
            min_samples_leaf: 1,
            feature_indices: vec![0, 1],
            sorted_features: vec![sorted_f0, sorted_f1],
        };
        let u = array![-1.0, -1.0, 1.0, 1.0];

        let update = state.fit_update(u.view(), None);
        println!("Update: {:?}", update);
        if let LearnerUpdate::Tree {
            node: TreeNode::Split { left, right, .. },
            ..
        } = update
        {
            assert!(matches!(*left, TreeNode::Leaf { value: v, .. } if v == -1.0));
            assert!(matches!(*right, TreeNode::Leaf { value: v, .. } if v == 1.0));
        } else {
            panic!("Expected Tree update");
        }
    }
}
