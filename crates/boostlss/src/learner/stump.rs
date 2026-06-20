use crate::learner::LearnerUpdate;
use ndarray::ArrayView1;

#[derive(Debug, Clone)]
pub struct Stump {
    pub feature_name: String,
}

impl Stump {
    pub fn new(feature_name: &str) -> Self {
        Self {
            feature_name: feature_name.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StumpFitState {
    pub sorted_x: Vec<(f64, usize)>,
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
        let state = StumpFitState { sorted_x: x };
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
