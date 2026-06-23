use ndarray::{s, Array2};

#[derive(Clone, Debug, PartialEq)]
pub struct BivariatePSpline {
    pub feature1: String,
    pub feature2: String,
    pub knots: usize,
    pub degree: usize,
    pub differences: usize,
    pub df: f64,
}

impl BivariatePSpline {
    pub fn new(feature1: &str, feature2: &str) -> Self {
        Self {
            feature1: feature1.to_string(),
            feature2: feature2.to_string(),
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
            slice.assign(&(b * val));
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
}
