//! Base-learners and cached factorization state.

pub mod bspatial;
pub mod linear;
pub use linear::Linear;

pub mod penalty;

pub mod pspline;
pub use pspline::PSpline;

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

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        match self {
            Self::Linear(l) => l.penalty_matrix(n_cols),
            Self::PSpline(p) => p.penalty_matrix(n_cols),
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
            let mut sorted_features = Vec::with_capacity(tree_learner.feature_indices.len());
            for &col_idx in &tree_learner.feature_indices {
                let col = data.design().column(col_idx);
                let mut sorted: Vec<(f64, usize)> = col
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, val)| (val, i))
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                sorted_features.push(sorted);
            }
            return Ok(LearnerFit::Tree(tree::TreeFitState {
                max_depth: tree_learner.max_depth,
                min_samples_leaf: tree_learner.min_samples_leaf,
                feature_indices: tree_learner.feature_indices.clone(),
                sorted_features,
            }));
        }

        if let Self::Stump(stump_learner) = self {
            let x = data.design().column(0); // Temporarily hardcoded for stump
            let mut sorted_x: Vec<(f64, usize)> = x
                .iter()
                .copied()
                .enumerate()
                .map(|(i, val)| (val, i))
                .collect();
            sorted_x.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            return Ok(LearnerFit::Stump(stump::StumpFitState {
                sorted_x,
                feature_idx: 0,
            }));
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
        let lambda = match self.target_df() {
            Some(df) => {
                let xtx = design.t().dot(&design);
                crate::learner::penalty::df_to_lambda(&xtx, &penalty, df)
            }
            None => 0.0,
        };

        let p = design.ncols();
        let xtx = design.t().dot(&design);
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
            (
                Self::Stump(state),
                LearnerUpdate::Stump {
                    split_val,
                    left_val,
                    right_val,
                },
            ) => {
                let x_col = data.design().column(state.feature_idx);
                x_col
                    .mapv(|val| {
                        if val <= *split_val {
                            *left_val
                        } else {
                            *right_val
                        }
                    })
                    .to_owned()
            }
            (Self::Tree(state), LearnerUpdate::Tree { node: root, .. }) => {
                let mut u_hat = ndarray::Array1::zeros(data.n_obs());
                for i in 0..u_hat.len() {
                    let mut node_ptr = root;
                    loop {
                        match node_ptr {
                            TreeNode::Leaf { value, .. } => {
                                u_hat[i] = *value;
                                break;
                            }
                            TreeNode::Split {
                                feature_idx,
                                threshold,
                                left,
                                right,
                            } => {
                                let col_idx = state.feature_indices[*feature_idx];
                                let val = data.design().column(col_idx)[i];
                                if val <= *threshold {
                                    node_ptr = left;
                                } else {
                                    node_ptr = right;
                                }
                            }
                        }
                    }
                }
                u_hat
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
