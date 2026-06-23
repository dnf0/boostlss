use crate::error::BoostlssError;
use crate::learner::PSpline;
use ndarray::{s, Array1, Array2};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BivariatePSpline {
    pub(crate) feature1_idx: usize,
    pub(crate) feature2_idx: usize,
    pub(crate) knots: usize,
    pub(crate) degree: usize,
    pub(crate) differences: usize,
    pub(crate) df: f64,
}

impl BivariatePSpline {
    pub fn new(feature1_idx: usize, feature2_idx: usize) -> Self {
        Self {
            feature1_idx,
            feature2_idx,
            knots: 20,
            degree: 3,
            differences: 2,
            df: 4.0,
        }
    }

    pub fn knots(mut self, knots: usize) -> Self {
        self.knots = knots;
        self
    }
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }
    pub fn differences(mut self, differences: usize) -> Self {
        self.differences = differences;
        self
    }
    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }

    pub fn build_design(
        &self,
        x1: &Array1<f64>,
        x2: &Array1<f64>,
    ) -> Result<Array2<f64>, BoostlssError> {
        let mut p1 = PSpline::new("")
            .with_knots(self.knots)
            .with_degree(self.degree)
            .with_differences(self.differences);
        let mut p2 = PSpline::new("")
            .with_knots(self.knots)
            .with_degree(self.degree)
            .with_differences(self.differences);

        let b1 = p1.build_design(x1)?;
        let b2 = p2.build_design(x2)?;

        let n_obs = b1.nrows();
        let p_cols1 = b1.ncols();
        let p_cols2 = b2.ncols();

        let mut design = Array2::zeros((n_obs, p_cols1 * p_cols2));
        for i in 0..n_obs {
            let row1 = b1.row(i);
            let row2 = b2.row(i);
            for j in 0..p_cols1 {
                for k in 0..p_cols2 {
                    design[[i, j * p_cols2 + k]] = row1[j] * row2[k];
                }
            }
        }
        Ok(design)
    }

    pub fn penalty_matrix(&self, p_cols1: usize, p_cols2: usize) -> Array2<f64> {
        let k1 = crate::learner::penalty::penalty_matrix(p_cols1, self.differences, false);
        let k2 = crate::learner::penalty::penalty_matrix(p_cols2, self.differences, false);

        let i1 = Array2::<f64>::eye(p_cols1);
        let i2 = Array2::<f64>::eye(p_cols2);

        kronecker_product(&k1, &i2) + kronecker_product(&i1, &k2)
    }
}

/// Computes the full Kronecker product of two 2D matrices
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let mut res = Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    for i in 0..a_rows {
        for j in 0..a_cols {
            let val = a[[i, j]];
            let mut slice = res.slice_mut(s![
                i * b_rows..(i + 1) * b_rows,
                j * b_cols..(j + 1) * b_cols
            ]);
            slice.zip_mut_with(b, |out, &b_val| *out = b_val * val);
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kronecker_product() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.5, 2.0], [3.0, 1.0]];
        let res = kronecker_product(&a, &b);
        let expected = array![
            [0.5, 2.0, 1.0, 4.0],
            [3.0, 1.0, 6.0, 2.0],
            [1.5, 6.0, 2.0, 8.0],
            [9.0, 3.0, 12.0, 4.0]
        ];
        assert_eq!(res, expected);
    }

    #[test]
    fn test_bspatial_fit_predict() {
        use crate::data::Dataset;
        use crate::family::GaussianLss;
        use crate::learner::{BaseLearner, LearnerUpdate};
        use crate::model::{Fitted, Scale, UpdateStep};
        use ndarray::Array1;

        let mut design = Array2::zeros((100, 2));
        let mut x1 = Array1::zeros(100);
        let mut x2 = Array1::zeros(100);
        for i in 0..10 {
            for j in 0..10 {
                x1[i * 10 + j] = i as f64 / 9.0;
                x2[i * 10 + j] = j as f64 / 9.0;
            }
        }
        design.column_mut(0).assign(&x1);
        design.column_mut(1).assign(&x2);
        let response = Array1::linspace(0., 1., 100);
        let ds = Dataset::new(design, response, None).unwrap();

        let mut learner: BaseLearner = BivariatePSpline::new(0, 1).knots(5).into();

        let u = Array1::ones(100);

        // This exercises initialize -> build_design and penalty_matrix
        let fit = learner.initialize(&Array1::zeros(100), &ds).unwrap();

        // Now get the update
        let update = fit.fit_update(u.view(), None);

        // Ensure the test checks predict using LearnerUpdate::Linear
        assert!(matches!(update, LearnerUpdate::Linear(_)));

        // Create a Fitted mock to test predict
        let mut fitted = Fitted::new(GaussianLss::new(), vec![0.0, 0.0], vec![(0, learner)]);
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update,
            risk_reduction: 0.1,
        });

        let pred = fitted.predict(&ds, "mu", Scale::Link).unwrap();
        assert_eq!(pred.len(), 100);
    }
}
