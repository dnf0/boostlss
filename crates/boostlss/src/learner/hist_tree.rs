use crate::data::Dataset;
use ndarray::Array2;
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
}
