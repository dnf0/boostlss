use crate::data::{DesignMatrix, SparseMatrix};
use crate::error::BoostlssError;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Note: Now uses sparse matrices (CSC) to efficiently handle categorical features
/// with many categories, avoiding previous OOM issues on high cardinality datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffects {
    pub feature_idx: usize,
    pub df: f64,
}

impl RandomEffects {
    pub fn new(feature_idx: usize) -> Self {
        Self {
            feature_idx,
            df: 4.0,
        }
    }

    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }

    pub fn build_design(&self, data: &crate::data::Dataset) -> Result<DesignMatrix, BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        let n_obs = col.len();
        if n_obs == 0 {
            return Ok(DesignMatrix::Dense(ndarray::Array2::zeros((0, 0))));
        }

        let mut max_idx = 0;
        for &val in col.iter() {
            if val.fract() != 0.0 || val < 0.0 {
                return Err(BoostlssError::DataError(
                    "RandomEffects requires non-negative integer indices".to_string(),
                ));
            }
            let idx = val as usize;
            if idx > max_idx {
                max_idx = idx;
            }
        }
        let n_cols = max_idx + 1;

        // Construct CSC matrix directly.
        let mut row_counts = vec![0; n_cols];
        for &val in col.iter() {
            row_counts[val as usize] += 1;
        }

        let mut indptr = Vec::with_capacity(n_cols + 1);
        indptr.push(0);
        let mut current = 0;
        for &count in row_counts.iter() {
            current += count;
            indptr.push(current);
        }

        let mut indices = vec![0; n_obs];
        let data_vals = vec![1.0; n_obs];

        let mut offsets = indptr.clone();
        for (row_idx, &val) in col.iter().enumerate() {
            let col_idx = val as usize;
            let offset = offsets[col_idx];
            indices[offset] = row_idx;
            offsets[col_idx] += 1;
        }

        let sparse = SparseMatrix::new(
            Array1::from_vec(data_vals),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (n_obs, n_cols),
        )?;

        Ok(DesignMatrix::Csc(sparse))
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> DesignMatrix {
        // Identity matrix as CSC
        let indptr: Vec<usize> = (0..=n_cols).collect();
        let indices: Vec<usize> = (0..n_cols).collect();
        let data_vals: Vec<f64> = vec![1.0; n_cols];

        let sparse = SparseMatrix::new(
            Array1::from_vec(data_vals),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (n_cols, n_cols),
        )
        .unwrap();

        DesignMatrix::Csc(sparse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_random_effects_design() {
        let re = RandomEffects::new(0);
        let x = array![[0.0], [2.0], [1.0], [0.0]];
        let y = array![0.0, 0.0, 0.0, 0.0];
        let data = crate::data::Dataset::new(x, y, None, None).unwrap();
        let design = match re.build_design(&data).unwrap() {
            DesignMatrix::Csc(d) => d,
            _ => panic!("Expected Csc"),
        };

        assert_eq!(design.shape, (4, 3));

        // Col 0 has row 0, 3
        // Col 1 has row 2
        // Col 2 has row 1
        assert_eq!(design.indptr.to_vec(), vec![0, 2, 3, 4]);
        assert_eq!(design.indices.to_vec(), vec![0, 3, 2, 1]);
        assert_eq!(design.data.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_random_effects_invalid_data() {
        let re = RandomEffects::new(0);
        let x1 = array![[-1.0], [0.0]];
        let y1 = array![0.0, 0.0];
        let data1 = crate::data::Dataset::new(x1, y1, None, None).unwrap();
        assert!(re.build_design(&data1).is_err());

        let x2 = array![[0.5], [1.0]];
        let y2 = array![0.0, 0.0];
        let data2 = crate::data::Dataset::new(x2, y2, None, None).unwrap();
        assert!(re.build_design(&data2).is_err());
    }

    #[test]
    fn test_random_effects_penalty() {
        let re = RandomEffects::new(0);
        let pen = match re.penalty_matrix(3) {
            DesignMatrix::Csc(d) => d,
            _ => panic!("Expected Csc"),
        };
        assert_eq!(pen.shape, (3, 3));
        assert_eq!(pen.indptr.to_vec(), vec![0, 1, 2, 3]);
        assert_eq!(pen.indices.to_vec(), vec![0, 1, 2]);
        assert_eq!(pen.data.to_vec(), vec![1.0, 1.0, 1.0]);
    }
}
