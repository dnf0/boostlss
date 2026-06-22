//! Base-learners and cached factorization state.

pub mod linear;
pub use linear::Linear;

pub mod penalty;

pub mod pspline;
pub use pspline::PSpline;

pub mod stump;
pub use stump::Stump;

pub mod tree;
pub use tree::{Tree, TreeNode};

#[derive(Debug, Clone)]
pub enum BaseLearner {
    Linear(Linear),
    PSpline(PSpline),
    Stump(Stump),
    Tree(Tree),
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

impl BaseLearner {
    pub fn build_design(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<Array2<f64>, crate::error::BoostlssError> {
        match self {
            Self::Linear(l) => l.build_design(x),
            Self::PSpline(p) => p.build_design(x),
            Self::Stump(_) => Err(crate::error::BoostlssError::DataError(
                "Stump does not use build_design".into(),
            )),
            Self::Tree(_) => Err(crate::error::BoostlssError::DataError(
                "Tree does not use build_design".into(),
            )),
        }
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        match self {
            Self::Linear(l) => l.penalty_matrix(n_cols),
            Self::PSpline(p) => p.penalty_matrix(n_cols),
            Self::Stump(_) => Array2::zeros((0, 0)),
            Self::Tree(_) => Array2::zeros((0, 0)),
        }
    }

    pub fn target_df(&self) -> Option<f64> {
        match self {
            Self::Linear(_) => None,
            Self::PSpline(p) => Some(p.df),
            Self::Stump(_) => None,
            Self::Tree(_) => None,
        }
    }
    pub fn initialize(
        &mut self,
        x: &Array1<f64>,
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
                sorted_features,
            }));
        }

        if let Self::Stump(_) = self {
            let mut sorted_x: Vec<(f64, usize)> = x
                .iter()
                .copied()
                .enumerate()
                .map(|(i, val)| (val, i))
                .collect();
            sorted_x.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            return Ok(LearnerFit::Stump(stump::StumpFitState { sorted_x }));
        }

        let design = self.build_design(x)?;
        let penalty = self.penalty_matrix(design.ncols());
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

#[derive(Debug, Clone)]
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
        let l = Linear::new("x");
        let bl: BaseLearner = l.into();
        assert!(matches!(bl, BaseLearner::Linear(_)));

        let p = PSpline::new("x");
        let bl: BaseLearner = p.into();
        assert!(matches!(bl, BaseLearner::PSpline(_)));

        let s = Stump::new("x");
        let bl: BaseLearner = s.into();
        assert!(matches!(bl, BaseLearner::Stump(_)));

        let t = Tree::new(vec![0]);
        let bl: BaseLearner = t.into();
        assert!(matches!(bl, BaseLearner::Tree(_)));
    }
}
