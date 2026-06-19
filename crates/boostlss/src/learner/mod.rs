//! Base-learners and cached factorization state.

use crate::error::BoostlssError;
use faer::linalg::solvers::Llt;
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1};

/// The fitted state for a base-learner.
/// Caches the Cholesky factor of (X^T X + lambda K) to make updates O(p^2) instead of O(p^3).
#[derive(Debug, Clone)]
pub struct LearnerFit {
    /// Accumulated coefficients
    pub coef: Array1<f64>,
    /// Precomputed X^T (for unweighted steps)
    pub xt: Array2<f64>,
    /// Cholesky factor L from faer. (L * L^T = A)
    pub chol_l: Mat<f64>,
    /// Number of times this learner was selected
    pub selected_count: usize,
}

impl LearnerFit {
    /// Factorize A = X^T X + \lambda K using faer's Cholesky decomposition.
    pub fn new(x: &Array2<f64>, penalty: &Array2<f64>, lambda: f64) -> Result<Self, BoostlssError> {
        let n = x.nrows();
        let p = x.ncols();

        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            for j in 0..p {
                for k in 0..p {
                    xtx[[j, k]] += x[[i, j]] * x[[i, k]];
                }
            }
        }

        let mut a = Mat::zeros(p, p);
        for j in 0..p {
            for k in 0..p {
                a[(j, k)] = xtx[[j, k]] + lambda * penalty[[j, k]];
            }
        }

        let llt = Llt::new(a.as_ref(), faer::Side::Lower).map_err(|_| {
            BoostlssError::DataError(
                "Cholesky decomposition failed: matrix not positive definite".to_string(),
            )
        })?;

        let mut chol_l = Mat::zeros(p, p);
        let l_ref = llt.L();
        for j in 0..p {
            for k in 0..=j {
                chol_l[(j, k)] = l_ref[(j, k)];
            }
        }

        let mut xt = Array2::<f64>::zeros((p, n));
        for j in 0..p {
            for i in 0..n {
                xt[[j, i]] = x[[i, j]];
            }
        }

        Ok(Self {
            coef: Array1::zeros(p),
            xt,
            chol_l,
            selected_count: 0,
        })
    }

    /// Solve (X^T X + \lambda K) beta = X^T u for the update step.
    pub fn solve_update(&self, u: ArrayView1<f64>) -> Array1<f64> {
        let p = self.xt.nrows();
        let n = self.xt.ncols();

        let mut xtu = Mat::zeros(p, 1);
        for j in 0..p {
            let mut sum = 0.0;
            for i in 0..n {
                sum += self.xt[[j, i]] * u[i];
            }
            xtu[(j, 0)] = sum;
        }

        // Solve L L^T beta = X^T u
        // First solve L y = X^T u in-place
        solve_lower_triangular_in_place(self.chol_l.as_ref(), xtu.as_mut(), faer::Par::Seq);
        // Then solve L^T beta = y in-place
        solve_upper_triangular_in_place(
            self.chol_l.as_ref().transpose(),
            xtu.as_mut(),
            faer::Par::Seq,
        );

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
        let beta = fit.solve_update(u.view());

        assert_eq!(beta.len(), 2);
    }
}
