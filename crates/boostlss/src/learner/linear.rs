use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Linear {
    pub col_name: String,
    pub intercept: bool,
}

impl Linear {
    pub fn new(col_name: &str) -> Self {
        Self {
            col_name: col_name.to_string(),
            intercept: true,
        }
    }

    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    pub fn build_design(&self, x: &Array1<f64>) -> Result<Array2<f64>, BoostlssError> {
        let n = x.len();
        if self.intercept {
            let mut xt = Array2::zeros((n, 2));
            for i in 0..n {
                xt[[i, 0]] = 1.0;
                xt[[i, 1]] = x[i];
            }
            Ok(xt)
        } else {
            let mut xt = Array2::zeros((n, 1));
            for i in 0..n {
                xt[[i, 0]] = x[i];
            }
            Ok(xt)
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
        let linear = Linear::new("x");
        let x = array![1.0, 2.0, 3.0];
        let design = linear.build_design(&x).unwrap();

        assert_eq!(design, array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
    }

    #[test]
    fn test_linear_without_intercept() {
        let linear = Linear::new("x").intercept(false);
        assert!(!linear.intercept);

        let x = array![1.0, 2.0, 3.0];
        let design = linear.build_design(&x).unwrap();

        assert_eq!(design, array![[1.0], [2.0], [3.0]]);
    }

    #[test]
    fn test_penalty_matrix() {
        let linear = Linear::new("x");
        let penalty = linear.penalty_matrix(2);
        assert_eq!(penalty, array![[0.0, 0.0], [0.0, 0.0]]);
    }
}
