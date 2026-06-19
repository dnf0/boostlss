//! Base-learners and cached factorization state.

pub mod linear;
pub use linear::Linear;

pub mod penalty;

pub enum BaseLearner {
    Linear(Linear),
}

use crate::error::BoostlssError;
use faer::linalg::solvers::Llt;
use faer::prelude::Solve;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1};

/// The fitted state for a base-learner.
/// Caches the Cholesky factor of (X^T X + lambda K) to make updates O(p^2) instead of O(p^3).
#[derive(Debug, Clone)]
pub struct LearnerFit {
    /// Accumulated coefficients
    pub coef: Array1<f64>,
    /// Cholesky factor L from faer. (L * L^T = A)
    pub llt: Llt<f64>,
    /// Number of times this learner was selected
    pub selected_count: usize,
}

impl LearnerFit {
    /// Factorize A = X^T X + \lambda K using faer's Cholesky decomposition.
    pub fn new(x: &Array2<f64>, penalty: &Array2<f64>, lambda: f64) -> Result<Self, BoostlssError> {
        let p = x.ncols();
        let xtx = x.t().dot(x);

        let a = Mat::from_fn(p, p, |j, k| xtx[[j, k]] + lambda * penalty[[j, k]]);

        let llt = Llt::new(a.as_ref(), faer::Side::Lower).map_err(|_| {
            BoostlssError::DataError(
                "Cholesky decomposition failed: matrix not positive definite".to_string(),
            )
        })?;

        Ok(Self {
            coef: Array1::zeros(p),
            llt,
            selected_count: 0,
        })
    }

    /// Solve (X^T X + \lambda K) beta = X^T u for the update step.
    pub fn solve_update(&self, x: &Array2<f64>, u: ArrayView1<f64>) -> Array1<f64> {
        let p = x.ncols();

        let xtu_nd = x.t().dot(&u);

        let mut xtu = Mat::from_fn(p, 1, |j, _| xtu_nd[j]);

        // Solve L L^T beta = X^T u in-place
        self.llt.solve_in_place(xtu.as_mut());

        Array1::from_shape_fn(p, |i| xtu[(i, 0)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_learner_fit() {
        let x = array![[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let lambda = 0.1;

        let fit = LearnerFit::new(&x, &penalty, lambda).unwrap();

        let u = array![1.0, 0.5, -0.5];
        let beta = fit.solve_update(&x, u.view());

        let expected_beta0 = 52.0 / 21.0;
        let expected_beta1 = -5.0 / 7.0;

        assert_eq!(beta.len(), 2);
        assert!((beta[0] - expected_beta0).abs() < 1e-8);
        assert!((beta[1] - expected_beta1).abs() < 1e-8);
    }
}
