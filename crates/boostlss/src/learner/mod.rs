//! Base-learners and cached factorization state.

pub mod bspatial;
pub mod linear;
pub use linear::Linear;

pub mod penalty;

pub mod spline_utils;

pub mod pspline;
pub use pspline::PSpline;

pub mod constrained_pspline;

pub mod stump;
pub use stump::Stump;

pub mod tree;
use serde::{Deserialize, Serialize};
pub use tree::{Tree, TreeNode};

pub mod random_effects;
pub use random_effects::RandomEffects;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseLearner {
    Linear(Linear),
    PSpline(PSpline),
    ConstrainedPSpline(constrained_pspline::ConstrainedPSpline),
    Stump(Stump),
    Tree(Tree),
    RandomEffects(RandomEffects),
    BivariatePSpline(bspatial::BivariatePSpline),
}

impl From<Linear> for BaseLearner {
    fn from(l: Linear) -> Self {
        Self::Linear(l)
    }
}

impl From<PSpline> for BaseLearner {
    fn from(p: PSpline) -> Self {
        Self::PSpline(p)
    }
}

impl From<constrained_pspline::ConstrainedPSpline> for BaseLearner {
    fn from(c: constrained_pspline::ConstrainedPSpline) -> Self {
        Self::ConstrainedPSpline(c)
    }
}

impl From<Stump> for BaseLearner {
    fn from(s: Stump) -> Self {
        Self::Stump(s)
    }
}

impl From<Tree> for BaseLearner {
    fn from(t: Tree) -> Self {
        Self::Tree(t)
    }
}

impl From<RandomEffects> for BaseLearner {
    fn from(r: RandomEffects) -> Self {
        Self::RandomEffects(r)
    }
}

impl From<bspatial::BivariatePSpline> for BaseLearner {
    fn from(b: bspatial::BivariatePSpline) -> Self {
        Self::BivariatePSpline(b)
    }
}

impl BaseLearner {
    pub fn build_design(
        &mut self,
        data: &crate::data::Dataset,
    ) -> Result<Array2<f64>, crate::error::BoostlssError> {
        match self {
            Self::Linear(l) => l.build_design(data),
            Self::PSpline(p) => p.build_design(data),
            Self::ConstrainedPSpline(c) => c.build_design(data),
            Self::RandomEffects(r) => r.build_design(data),
            Self::Stump(_) => Err(crate::error::BoostlssError::DataError(
                "Stump does not use build_design".into(),
            )),
            Self::Tree(_) => Err(crate::error::BoostlssError::DataError(
                "Tree does not use build_design".into(),
            )),
            Self::BivariatePSpline(_) => Err(crate::error::BoostlssError::DataError(
                "BivariatePSpline does not use build_design(x)".into(),
            )),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Linear(l) => format!("Linear_{}", l.feature_idx),
            Self::PSpline(p) => format!("PSpline_{}", p.feature_idx),
            Self::ConstrainedPSpline(c) => format!("ConstrainedPSpline_{}", c.feature_idx),
            Self::RandomEffects(r) => format!("RandomEffects_{}", r.feature_idx),
            Self::Stump(s) => format!("Stump_{}", s.feature_idx),
            Self::Tree(_) => "Tree".to_string(),
            Self::BivariatePSpline(bp) => {
                format!("BivariatePSpline_{}_{}", bp.feature1_idx, bp.feature2_idx)
            }
        }
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        match self {
            Self::Linear(l) => l.penalty_matrix(n_cols),
            Self::PSpline(p) => p.penalty_matrix(n_cols),
            Self::ConstrainedPSpline(c) => c.penalty_matrix(n_cols),
            Self::RandomEffects(r) => r.penalty_matrix(n_cols),
            Self::Stump(_) => Array2::zeros((0, 0)),
            Self::Tree(_) => Array2::zeros((0, 0)),
            Self::BivariatePSpline(_) => Array2::zeros((0, 0)),
        }
    }

    pub fn target_df(&self) -> Option<f64> {
        match self {
            Self::Linear(_) => None,
            Self::PSpline(p) => Some(p.df),
            Self::ConstrainedPSpline(c) => Some(c.df),
            Self::RandomEffects(r) => Some(r.df),
            Self::Stump(_) => None,
            Self::Tree(_) => None,
            Self::BivariatePSpline(bp) => Some(bp.df),
        }
    }
    pub fn initialize(
        &mut self,
        data: &crate::data::Dataset,
    ) -> Result<LearnerFit, crate::error::BoostlssError> {
        if let Self::Tree(tree_learner) = self {
            return Ok(LearnerFit::Tree(tree_learner.build_fit_state(data)?));
        }

        if let Self::Stump(stump_learner) = self {
            return Ok(LearnerFit::Stump(stump_learner.build_fit_state(data)?));
        }

        let (design, penalty) = match self {
            Self::BivariatePSpline(bp) => {
                let design = bp.build_design(data)?;
                let penalty = bp.penalty_matrix(bp.knots + bp.degree + 1, bp.knots + bp.degree + 1);
                (design, penalty)
            }
            _ => {
                let d = self.build_design(data)?;
                let p = self.penalty_matrix(d.ncols());
                (d, p)
            }
        };
        let mut xtx = design.t().dot(&design);
        if let Some(w) = data.weights() {
            let mut weighted_design = design.clone();
            for i in 0..design.nrows() {
                let wi = w[i];
                for j in 0..design.ncols() {
                    weighted_design[[i, j]] *= wi;
                }
            }
            xtx = design.t().dot(&weighted_design);
        }

        let lambda = match self.target_df() {
            Some(df) => crate::learner::penalty::df_to_lambda(&xtx, &penalty, df),
            None => 0.0,
        };

        if let Self::ConstrainedPSpline(cp) = self {
            let p = design.ncols();
            let order = match cp.constraint {
                constrained_pspline::Constraint::MonotonicIncreasing
                | constrained_pspline::Constraint::MonotonicDecreasing => 1,
                constrained_pspline::Constraint::Convex
                | constrained_pspline::Constraint::Concave => 2,
            };
            let constraint_diff = penalty::difference_matrix(p, order, false);
            return Ok(LearnerFit::ConstrainedPSpline(
                constrained_pspline::ConstrainedFitState {
                    xtx,
                    design,
                    smooth_penalty: penalty,
                    lambda_smooth: lambda,
                    constraint_diff,
                    constraint: cp.constraint.clone(),
                    max_iter: cp.max_iter,
                    tolerance: cp.tolerance,
                },
            ));
        }

        let p = design.ncols();
        let a = faer::Mat::from_fn(p, p, |j, k| xtx[[j, k]] + lambda * penalty[[j, k]]);
        let llt = faer::linalg::solvers::Llt::new(a.as_ref(), faer::Side::Lower).map_err(|_| {
            crate::error::BoostlssError::DataError(
                "Cholesky decomposition failed: matrix not positive definite".to_string(),
            )
        })?;

        Ok(LearnerFit::Linear(LinearFitState {
            coef: Array1::zeros(p),
            llt,
            design,
        }))
    }
}

use faer::linalg::solvers::Llt;
use faer::prelude::Solve;
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearnerUpdate {
    Linear(Array1<f64>),
    Stump {
        split_val: f64,
        left_val: f64,
        right_val: f64,
    },
    Tree {
        node: TreeNode,
        param: String,
    },
}

impl LearnerUpdate {
    pub fn scale(&mut self, factor: f64) {
        match self {
            Self::Linear(coef) => {
                coef.mapv_inplace(|v| v * factor);
            }
            Self::Stump {
                left_val,
                right_val,
                ..
            } => {
                *left_val *= factor;
                *right_val *= factor;
            }
            Self::Tree { node, .. } => {
                node.scale(factor);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearFitState {
    pub coef: Array1<f64>,
    pub llt: Llt<f64>,
    pub design: Array2<f64>,
}

#[derive(Debug, Clone)]
pub enum LearnerFit {
    Linear(LinearFitState),
    ConstrainedPSpline(constrained_pspline::ConstrainedFitState),
    Stump(stump::StumpFitState),
    Tree(tree::TreeFitState),
}

impl LearnerFit {
    pub fn fit_update(
        &self,
        u: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> LearnerUpdate {
        match self {
            Self::ConstrainedPSpline(state) => state.fit_update(u, weights),
            Self::Linear(state) => {
                let p = state.design.ncols();
                let xtu_nd = state.design.t().dot(&u);
                let mut xtu = faer::Mat::from_fn(p, 1, |j, _| xtu_nd[j]);
                state.llt.solve_in_place(xtu.as_mut());
                LearnerUpdate::Linear(Array1::from_shape_fn(p, |i| xtu[(i, 0)]))
            }
            Self::Stump(state) => state.fit_update(u, weights),
            Self::Tree(state) => state.fit_update(u, weights),
        }
    }

    pub fn predict_update(
        &self,
        update: &LearnerUpdate,
        data: &crate::data::Dataset,
    ) -> Array1<f64> {
        match (self, update) {
            (Self::Linear(state), LearnerUpdate::Linear(coef)) => state.design.dot(coef),
            (Self::ConstrainedPSpline(state), LearnerUpdate::Linear(coef)) => {
                state.design.dot(coef)
            }
            (
                Self::Stump(state),
                LearnerUpdate::Stump {
                    split_val,
                    left_val,
                    right_val,
                },
            ) => {
                let stump_learner = Stump::new(state.feature_idx);
                stump_learner
                    .predict(*split_val, *left_val, *right_val, data)
                    .unwrap_or_else(|_| ndarray::Array1::zeros(data.n_obs()))
            }
            (Self::Tree(state), LearnerUpdate::Tree { node: root, .. }) => {
                let tree_learner = Tree::new(state.feature_indices.clone());
                tree_learner
                    .predict(root, data)
                    .unwrap_or_else(|_| ndarray::Array1::zeros(data.n_obs()))
            }
            _ => unreachable!("LearnerFit and LearnerUpdate types must match"),
        }
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

        let p = x.ncols();
        let xtx = x.t().dot(&x);
        let a = faer::Mat::from_fn(p, p, |j, k| xtx[[j, k]] + lambda * penalty[[j, k]]);
        let llt = faer::linalg::solvers::Llt::new(a.as_ref(), faer::Side::Lower).unwrap();

        let fit = LearnerFit::Linear(LinearFitState {
            coef: Array1::zeros(p),
            llt,
            design: x,
        });

        let u = array![1.0, 0.5, -0.5];
        let beta = match fit.fit_update(u.view(), None) {
            LearnerUpdate::Linear(b) => b,
            _ => panic!("Expected Linear update"),
        };

        let expected_beta0 = 52.0 / 21.0;
        let expected_beta1 = -5.0 / 7.0;

        assert_eq!(beta.len(), 2);
        assert!((beta[0] - expected_beta0).abs() < 1e-8);
        assert!((beta[1] - expected_beta1).abs() < 1e-8);
    }

    #[test]
    fn test_from_impls() {
        let l = Linear::new(0);
        let bl: BaseLearner = l.into();
        assert!(matches!(bl, BaseLearner::Linear(_)));

        let p = PSpline::new(0);
        let bl: BaseLearner = p.into();
        assert!(matches!(bl, BaseLearner::PSpline(_)));

        let cp = constrained_pspline::ConstrainedPSpline::new(
            0,
            constrained_pspline::Constraint::MonotonicIncreasing,
        );
        let bl: BaseLearner = cp.into();
        assert!(matches!(bl, BaseLearner::ConstrainedPSpline(_)));

        let s = Stump::new(0);
        let bl: BaseLearner = s.into();
        assert!(matches!(bl, BaseLearner::Stump(_)));

        let t = Tree::new(vec![0]);
        let bl: BaseLearner = t.into();
        assert!(matches!(bl, BaseLearner::Tree(_)));
    }

    #[test]
    fn test_from_impls_random_effects() {
        let r = RandomEffects::new(0);
        let bl: BaseLearner = r.into();
        assert!(matches!(bl, BaseLearner::RandomEffects(_)));
    }
}
