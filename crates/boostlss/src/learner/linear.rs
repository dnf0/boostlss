use crate::error::BoostlssError;
use ndarray::Array2;
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

    pub fn build_design(
        &self,
        data: &crate::data::Dataset,
    ) -> Result<crate::data::DesignMatrix, BoostlssError> {
        let n_obs = data.n_obs();
        let n_cols = if self.intercept { 2 } else { 1 };
        let mut design = Array2::zeros((n_obs, n_cols));
        let col = data.design().get_column(self.feature_idx)?;
        let mut offset = 0;
        if self.intercept {
            design.column_mut(0).fill(1.0);
            offset = 1;
        }
        design.column_mut(offset).assign(&col);
        Ok(crate::data::DesignMatrix::Dense(design))
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> crate::data::DesignMatrix {
        crate::data::DesignMatrix::Dense(Array2::zeros((n_cols, n_cols)))
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
        let data = crate::data::Dataset::new(x, y, None, None).unwrap();
        let design = match linear.build_design(&data).unwrap() {
            crate::data::DesignMatrix::Dense(d) => d,
            _ => panic!("Expected Dense"),
        };

        assert_eq!(design, array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
    }

    #[test]
    fn test_linear_without_intercept() {
        let linear = Linear::new(0).intercept(false);
        assert!(!linear.intercept);

        let x = array![[1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None, None).unwrap();
        let design = match linear.build_design(&data).unwrap() {
            crate::data::DesignMatrix::Dense(d) => d,
            _ => panic!("Expected Dense"),
        };

        assert_eq!(design, array![[1.0], [2.0], [3.0]]);
    }

    #[test]
    fn test_penalty_matrix() {
        let linear = Linear::new(0);
        let penalty = match linear.penalty_matrix(2) {
            crate::data::DesignMatrix::Dense(d) => d,
            _ => panic!("Expected Dense"),
        };
        assert_eq!(penalty, array![[0.0, 0.0], [0.0, 0.0]]);
    }
}

#[cfg(test)]
mod tests_new {
    use super::*;
    use crate::data::Dataset;
    use ndarray::array;

    #[test]
    fn test_linear_extracts_correct_column() {
        // Dataset with 2 features
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = Dataset::new(x, y, None, None).unwrap();

        // Linear learner on feature_idx = 1
        let linear = Linear::new(1).intercept(false);

        let design = match linear.build_design(&data).unwrap() {
            crate::data::DesignMatrix::Dense(d) => d,
            _ => panic!("Expected Dense"),
        };
        assert_eq!(design, array![[10.0], [20.0], [30.0]]);
    }
}
