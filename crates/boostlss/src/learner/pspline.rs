use crate::error::BoostlssError;
use crate::learner::penalty::penalty_matrix;
use crate::learner::spline_utils::{build_bspline_design, SplineData};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PSpline {
    pub feature_idx: usize,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub is_cyclic: bool,
    pub df: f64,
    pub spline_data: Option<SplineData>,
}

impl PSpline {
    pub fn new(feature_idx: usize) -> Self {
        Self {
            feature_idx,
            knots: 20,
            degree: 3,
            differences: 2,
            is_cyclic: false,
            df: 4.0,
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

    pub fn cyclic(mut self, cyclic: bool) -> Self {
        self.is_cyclic = cyclic;
        self
    }

    /// Cox-de Boor recursion for evaluating B-spline basis functions.
    pub fn build_design(
        &mut self,
        data: &crate::data::Dataset,
    ) -> Result<Array2<f64>, BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        // use col instead of data.design().column(self.feature_idx)
        let b = build_bspline_design(&col, self.knots, self.degree, &mut self.spline_data)?;

        if self.is_cyclic {
            let p = self.knots + self.degree + 1;
            let c = self.knots + 1;
            let mut b_cyclic = Array2::zeros((b.nrows(), c));

            for i in 0..b.nrows() {
                for j in 0..c {
                    b_cyclic[[i, j]] = b[[i, j]];
                }
                for j in c..p {
                    b_cyclic[[i, j - c]] += b[[i, j]];
                }
            }
            return Ok(b_cyclic);
        }

        Ok(b)
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        penalty_matrix(n_cols, self.differences, self.is_cyclic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pspline_new() {
        let ps = PSpline::new(0);
        assert_eq!(ps.feature_idx, 0);
        assert_eq!(ps.knots, 20);
        assert_eq!(ps.degree, 3);
    }

    #[test]
    fn test_pspline_cyclic_builder() {
        let ps = PSpline::new(0).cyclic(true);
        assert!(ps.is_cyclic);
    }

    #[test]
    fn test_pspline_build_design() {
        let mut ps = PSpline::new(0);
        let x = array![[0.0], [0.5], [1.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None, None).unwrap();
        let design = ps.build_design(&data).unwrap();

        let p = ps.knots + ps.degree + 1;
        assert_eq!(design.shape(), &[3, p]);
    }

    #[test]
    fn test_pspline_build_design_cyclic() {
        let mut ps = PSpline::new(0).with_knots(5).with_degree(3).cyclic(true);
        let x = array![[0.0], [0.5], [1.0]];
        let y = array![0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None, None).unwrap();
        let design = ps.build_design(&data).unwrap();

        // Standard dimension is knots + degree + 1 (5 + 3 + 1 = 9)
        // Cyclic dimension drops the rightmost degree columns: knots + 1 = 6
        assert_eq!(design.shape(), &[3, 6]);

        // Ensure values sum to 1 row-wise (partition of unity)
        for i in 0..3 {
            let sum: f64 = design.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
