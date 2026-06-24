use crate::error::BoostlssError;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub feature_idx: usize,
    intercept: bool,
}

impl Linear {
    pub fn new(feature_idx: usize) -> Self {
        Self {
            feature_idx,
            intercept: true,
        }
    }

    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    pub fn build_design(&self, data: &crate::data::Dataset) -> Result<Array2<f64>, BoostlssError> {
        let x = data.design().column(self.feature_idx);
        let n = x.len();
        if self.intercept {
            let mut xt = Array2::ones((n, 2));
            xt.column_mut(1).assign(&x);
            Ok(xt)
        } else {
            Ok(x.to_owned()
                .into_shape_with_order((n, 1))
                .expect("Shape is guaranteed to match"))
        }
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        Array2::zeros((n_cols, n_cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_with_intercept() {
        let linear = Linear::new(0);
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None).unwrap();
        let design = linear.build_design(&data).unwrap();

        assert_eq!(design, array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
    }

    #[test]
    fn test_linear_without_intercept() {
        let linear = Linear::new(0).intercept(false);
        assert!(!linear.intercept);

        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None).unwrap();
        let design = linear.build_design(&data).unwrap();

        assert_eq!(design, array![[1.0], [2.0], [3.0]]);
    }

    #[test]
    fn test_penalty_matrix() {
        let linear = Linear::new(0);
        let penalty = linear.penalty_matrix(2);
        assert_eq!(penalty, array![[0.0, 0.0], [0.0, 0.0]]);
    }
}

#[cfg(test)]
mod tests_new {
    use super::*;
    use crate::data::Dataset;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_linear_extracts_correct_column() {
        // Dataset with 2 features
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = Dataset::new(x, y, None).unwrap();

        // Linear learner on feature_idx = 1
        let mut linear = Linear::new(1).intercept(false);

        let design = linear.build_design(&data).unwrap();
        assert_eq!(design, array![[10.0], [20.0], [30.0]]);
    }
}
