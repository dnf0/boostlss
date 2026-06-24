// crates/boostlss/src/learner/constrained_pspline.rs
use crate::error::BoostlssError;
use crate::learner::penalty::penalty_matrix;
use crate::learner::spline_utils::{build_bspline_design, SplineData};
use crate::learner::LearnerUpdate;
use faer::linalg::solvers::Llt;
use faer::prelude::Solve;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    MonotonicIncreasing,
    MonotonicDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedPSpline {
    pub feature_idx: usize,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub df: f64,
    pub constraint: Constraint,
    pub max_iter: usize,
    pub tolerance: f64,
    pub spline_data: Option<SplineData>,
}

impl ConstrainedPSpline {
    pub fn new(feature_idx: usize, constraint: Constraint) -> Self {
        Self {
            feature_idx,
            knots: 20,
            degree: 3,
            differences: 2, // Smoothness differences
            df: 4.0,
            constraint,
            max_iter: 10,
            tolerance: 1e-6,
            spline_data: None,
        }
    }

    pub fn with_knots(mut self, knots: usize) -> Self {
        self.knots = knots;
        self
    }
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }
    pub fn with_differences(mut self, differences: usize) -> Self {
        self.differences = differences;
        self
    }
    pub fn with_df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn build_design(
        &mut self,
        data: &crate::data::Dataset,
    ) -> Result<Array2<f64>, BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        // use col instead of data.design().column(self.feature_idx)
        build_bspline_design(&col, self.knots, self.degree, &mut self.spline_data)
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        penalty_matrix(n_cols, self.differences, false)
    }
}

// State used during IRLS
#[derive(Debug, Clone)]
pub struct ConstrainedFitState {
    pub xtx: Array2<f64>,
    pub design: Array2<f64>,
    pub smooth_penalty: Array2<f64>,
    pub lambda_smooth: f64,
    pub constraint_diff: Array2<f64>, // D matrix for constraint
    pub constraint: Constraint,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl ConstrainedFitState {
    pub fn fit_update(
        &self,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> LearnerUpdate {
        let p = self.design.ncols();
        let kappa = 1e6; // Large penalty for violating constraints

        // Initial unconstrained solve. Support weights if provided.
        let xtu_nd = match weights {
            Some(w) => {
                let weighted_u = &u * &w;
                self.design.t().dot(&weighted_u)
            }
            None => self.design.t().dot(&u),
        };

        let xtu = Mat::from_fn(p, 1, |j, _| xtu_nd[j]);
        let mut a = Mat::from_fn(p, p, |j, k| {
            self.xtx[[j, k]] + self.lambda_smooth * self.smooth_penalty[[j, k]]
        });

        let mut llt = Llt::new(a.as_ref(), faer::Side::Lower).unwrap();
        let mut beta_faer = llt.solve(xtu.as_ref());
        let mut beta = Array1::from_shape_fn(p, |i| beta_faer[(i, 0)]);

        for _ in 0..self.max_iter {
            let mut v = Array1::zeros(self.constraint_diff.nrows());
            let diffs = self.constraint_diff.dot(&beta);

            for i in 0..v.len() {
                v[i] = match self.constraint {
                    Constraint::MonotonicIncreasing => {
                        if diffs[i] < 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Constraint::MonotonicDecreasing => {
                        if diffs[i] > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Constraint::Convex => {
                        if diffs[i] < 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Constraint::Concave => {
                        if diffs[i] > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
            }

            // Check if constraints are met
            if v.sum() == 0.0 {
                break;
            }

            // V_mat = D^T * diag(v) * D
            let mut dt_v_d: Array2<f64> = Array2::zeros((p, p));
            for i in 0..self.constraint_diff.nrows() {
                if v[i] > 0.0 {
                    let row = self.constraint_diff.row(i);
                    for j in 0..p {
                        for k in 0..p {
                            dt_v_d[[j, k]] += row[j] * row[k];
                        }
                    }
                }
            }

            a = Mat::from_fn(p, p, |j, k| {
                self.xtx[[j, k]]
                    + self.lambda_smooth * self.smooth_penalty[[j, k]]
                    + kappa * dt_v_d[[j, k]]
            });

            llt = Llt::new(a.as_ref(), faer::Side::Lower).unwrap();
            beta_faer = llt.solve(xtu.as_ref());

            let mut max_diff = 0.0;
            let mut next_beta = Array1::zeros(p);
            for i in 0..p {
                next_beta[i] = beta_faer[(i, 0)];
                let diff = (next_beta[i] - beta[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }

            beta = next_beta;
            if max_diff < self.tolerance {
                break;
            }
        }

        LearnerUpdate::Linear(beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_constrained_pspline_new() {
        let ps = ConstrainedPSpline::new(0, Constraint::MonotonicIncreasing);
        assert_eq!(ps.feature_idx, 0);
        assert_eq!(ps.constraint, Constraint::MonotonicIncreasing);
    }

    #[test]
    fn test_constrained_pspline_build_design() {
        let mut ps = ConstrainedPSpline::new(0, Constraint::MonotonicIncreasing);
        let x = array![[0.0], [0.5], [1.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None).unwrap();
        let design = ps.build_design(&data).unwrap();
        let p = ps.knots + ps.degree + 1;
        assert_eq!(design.shape(), &[3, p]);
    }
}
