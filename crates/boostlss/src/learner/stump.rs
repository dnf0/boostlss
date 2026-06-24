use crate::learner::LearnerUpdate;
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stump {
    pub feature_idx: usize,
}

impl Stump {
    pub fn new(feature_idx: usize) -> Self {
        Self { feature_idx }
    }

    pub fn build_fit_state(
        &self,
        data: &crate::data::Dataset,
    ) -> Result<StumpFitState, crate::error::BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        // use col instead of data.design().column(self.feature_idx)
        let mut sorted_x: Vec<(f64, usize)> = col
            .iter()
            .copied()
            .enumerate()
            .map(|(i, val)| (val, i))
            .collect();
        sorted_x.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(StumpFitState {
            sorted_x,
            feature_idx: self.feature_idx,
        })
    }

    pub fn predict(
        &self,
        split_val: f64,
        left_val: f64,
        right_val: f64,
        data: &crate::data::Dataset,
    ) -> Result<ndarray::Array1<f64>, crate::error::BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        // use col instead of data.design().column(self.feature_idx)
        Ok(col.mapv(|val| {
            if val <= split_val {
                left_val
            } else {
                right_val
            }
        }))
    }
}

#[derive(Debug, Clone)]
pub struct StumpFitState {
    pub sorted_x: Vec<(f64, usize)>,
    pub feature_idx: usize,
}

impl StumpFitState {
    pub fn fit_update(
        &self,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> LearnerUpdate {
        let n = self.sorted_x.len();
        if n == 0 {
            return LearnerUpdate::Stump {
                split_val: 0.0,
                left_val: 0.0,
                right_val: 0.0,
            };
        }

        let total_w: f64 = match weights {
            Some(w) => w.sum(),
            None => n as f64,
        };

        let total_wu: f64 = match weights {
            Some(w) => (&w * &u).sum(),
            None => u.sum(),
        };

        let mut left_w = 0.0;
        let mut left_wu = 0.0;
        let mut best_score = f64::NEG_INFINITY;

        let mut best_split_val = self.sorted_x[0].0;
        let mut best_left_val = if total_w > 0.0 {
            total_wu / total_w
        } else {
            0.0
        };
        let mut best_right_val = best_left_val;

        for i in 0..(n - 1) {
            let (val, idx) = self.sorted_x[i];
            let w_i = weights.as_ref().map_or(1.0, |w| w[idx]);
            let u_i = u[idx];

            left_w += w_i;
            left_wu += w_i * u_i;

            let next_val = self.sorted_x[i + 1].0;
            if (val - next_val).abs() < f64::EPSILON {
                continue; // Only split between distinct values
            }

            let right_w = total_w - left_w;
            let right_wu = total_wu - left_wu;

            if left_w <= 1e-10 || right_w <= 1e-10 {
                continue;
            }

            let score = (left_wu * left_wu) / left_w + (right_wu * right_wu) / right_w;

            if score > best_score {
                best_score = score;
                best_split_val = (val + next_val) / 2.0;
                best_left_val = left_wu / left_w;
                best_right_val = right_wu / right_w;
            }
        }

        LearnerUpdate::Stump {
            split_val: best_split_val,
            left_val: best_left_val,
            right_val: best_right_val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_stump_fit() {
        let x = vec![(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3)];
        let state = StumpFitState {
            sorted_x: x,
            feature_idx: 0,
        };
        let u = array![-1.0, -1.0, 1.0, 1.0];

        let update = state.fit_update(u.view(), None);
        if let LearnerUpdate::Stump {
            split_val,
            left_val,
            right_val,
        } = update
        {
            assert!((2.0..3.0).contains(&split_val));
            assert_eq!(left_val, -1.0);
            assert_eq!(right_val, 1.0);
        } else {
            panic!("Expected Stump update");
        }
    }
}
