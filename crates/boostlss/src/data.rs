use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub data: Array1<f64>,
    pub indices: Array1<usize>,
    pub indptr: Array1<usize>,
    pub shape: (usize, usize),
}

impl SparseMatrix {
    pub fn new(
        data: Array1<f64>,
        indices: Array1<usize>,
        indptr: Array1<usize>,
        shape: (usize, usize),
    ) -> Result<Self, BoostlssError> {
        if data.len() != indices.len() {
            return Err(BoostlssError::DataError(
                "Data and indices must have the same length".to_string(),
            ));
        }
        if indptr.is_empty() {
            return Err(BoostlssError::DataError(
                "indptr must not be empty".to_string(),
            ));
        }

        let expected_csr = shape.0 + 1;
        let expected_csc = shape.1 + 1;
        if indptr.len() != expected_csr && indptr.len() != expected_csc {
            return Err(BoostlssError::DataError(
                "indptr length does not match expected length for CSR or CSC format".to_string(),
            ));
        }

        if indptr[indptr.len() - 1] != data.len() {
            return Err(BoostlssError::DataError(
                "Last element of indptr must equal the number of non-zero elements".to_string(),
            ));
        }

        Ok(Self {
            data,
            indices,
            indptr,
            shape,
        })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum DesignMatrix {
    Dense(Array2<f64>),
    Csr(SparseMatrix),
    Csc(SparseMatrix),
}

impl DesignMatrix {
    pub fn get_column(&self, col_idx: usize) -> Result<Array1<f64>, BoostlssError> {
        if col_idx >= self.ncols() {
            return Err(BoostlssError::DataError(
                "Column index out of bounds".to_string(),
            ));
        }

        match self {
            Self::Dense(mat) => Ok(mat.column(col_idx).to_owned()),
            Self::Csc(sparse) => {
                let mut col = Array1::zeros(sparse.shape.0);
                let start = sparse.indptr[col_idx];
                let end = sparse.indptr[col_idx + 1];
                for i in start..end {
                    let row_idx = sparse.indices[i];
                    col[row_idx] = sparse.data[i];
                }
                Ok(col)
            }
            Self::Csr(sparse) => {
                let mut col = Array1::zeros(sparse.shape.0);
                for row_idx in 0..sparse.shape.0 {
                    let start = sparse.indptr[row_idx];
                    let end = sparse.indptr[row_idx + 1];
                    for i in start..end {
                        if sparse.indices[i] == col_idx {
                            col[row_idx] = sparse.data[i];
                            break;
                        }
                    }
                }
                Ok(col)
            }
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.nrows(),
            Self::Csr(sparse) => sparse.shape.0,
            Self::Csc(sparse) => sparse.shape.0,
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.ncols(),
            Self::Csr(sparse) => sparse.shape.1,
            Self::Csc(sparse) => sparse.shape.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    design: DesignMatrix,
    response: Array1<f64>,
    weights: Option<Array1<f64>>,
    censoring: Option<Array1<bool>>,
}

impl Dataset {
    fn validate_and_create(
        design: DesignMatrix,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
        censoring: Option<Array1<bool>>,
    ) -> Result<Self, BoostlssError> {
        let n = design.nrows();
        if response.len() != n {
            return Err(BoostlssError::DataError(format!(
                "Design has {} rows, but response has length {}",
                n,
                response.len()
            )));
        }
        if let Some(w) = &weights {
            if w.len() != n {
                return Err(BoostlssError::DataError(format!(
                    "Design has {} rows, but weights have length {}",
                    n,
                    w.len()
                )));
            }
            if w.iter().any(|&wi| wi < 0.0) {
                return Err(BoostlssError::DataError(
                    "Weights cannot be negative".into(),
                ));
            }
        }
        if let Some(c) = &censoring {
            if c.len() != n {
                return Err(BoostlssError::DataError(format!(
                    "Design has {} rows, but censoring has length {}",
                    n,
                    c.len()
                )));
            }
        }
        Ok(Self {
            design,
            response,
            weights,
            censoring,
        })
    }

    pub fn new(
        design: Array2<f64>,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
        censoring: Option<Array1<bool>>,
    ) -> Result<Self, BoostlssError> {
        Self::validate_and_create(DesignMatrix::Dense(design), response, weights, censoring)
    }

    pub fn new_csr(
        sparse: SparseMatrix,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
        censoring: Option<Array1<bool>>,
    ) -> Result<Self, BoostlssError> {
        Self::validate_and_create(DesignMatrix::Csr(sparse), response, weights, censoring)
    }

    pub fn new_csc(
        sparse: SparseMatrix,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
        censoring: Option<Array1<bool>>,
    ) -> Result<Self, BoostlssError> {
        Self::validate_and_create(DesignMatrix::Csc(sparse), response, weights, censoring)
    }

    pub fn design(&self) -> &DesignMatrix {
        &self.design
    }

    pub fn n_obs(&self) -> usize {
        self.design.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.design.ncols()
    }

    pub fn response(&self) -> &Array1<f64> {
        &self.response
    }

    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }

    pub fn censoring(&self) -> Option<&Array1<bool>> {
        self.censoring.as_ref()
    }

    pub fn set_censoring(&mut self, censoring: Array1<bool>) -> Result<(), BoostlssError> {
        if censoring.len() != self.n_obs() {
            return Err(BoostlssError::DataError(format!(
                "Design has {} rows, but censoring has length {}",
                self.n_obs(),
                censoring.len()
            )));
        }
        self.censoring = Some(censoring);
        Ok(())
    }

    pub fn set_weights(&mut self, weights: Array1<f64>) -> Result<(), BoostlssError> {
        if weights.len() != self.n_obs() {
            return Err(BoostlssError::DataError(format!(
                "Design has {} rows, but weights have length {}",
                self.n_obs(),
                weights.len()
            )));
        }
        if weights.iter().any(|&wi| wi < 0.0) {
            return Err(BoostlssError::DataError(
                "Weights cannot be negative".into(),
            ));
        }
        self.weights = Some(weights);
        Ok(())
    }

    pub fn with_weights(&self, new_weights: Array1<f64>) -> Result<Self, BoostlssError> {
        let n = self.n_obs();
        if new_weights.len() != n {
            return Err(BoostlssError::DataError(format!(
                "Design has {} rows, but new weights have length {}",
                n,
                new_weights.len()
            )));
        }
        if new_weights.iter().any(|&wi| wi < 0.0) {
            return Err(BoostlssError::DataError(
                "Weights cannot be negative".into(),
            ));
        }

        let combined_weights = if let Some(existing) = &self.weights {
            existing * &new_weights
        } else {
            new_weights
        };

        Ok(Self {
            design: self.design.clone(),
            response: self.response.clone(),
            weights: Some(combined_weights),
            censoring: self.censoring.clone(),
        })
    }

    pub fn subset(&self, indices: &[usize]) -> Result<Self, BoostlssError> {
        let DesignMatrix::Dense(mat) = &self.design else {
            return Err(BoostlssError::DataError(
                "Subset only supported for dense matrices".to_string(),
            ));
        };
        let n = indices.len();
        let mut new_design = ndarray::Array2::zeros((n, mat.ncols()));
        let mut new_response = ndarray::Array1::zeros(n);
        let mut new_weights = self.weights.as_ref().map(|_| ndarray::Array1::zeros(n));
        let mut new_censoring = self
            .censoring
            .as_ref()
            .map(|_| ndarray::Array1::from_elem(n, false));

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= mat.nrows() {
                return Err(BoostlssError::DataError("Index out of bounds".to_string()));
            }
            new_design.row_mut(i).assign(&mat.row(idx));
            new_response[i] = self.response[idx];
            if let (Some(ref mut w), Some(ref old_w)) = (&mut new_weights, &self.weights) {
                w[i] = old_w[idx];
            }
            if let (Some(ref mut c), Some(ref old_c)) = (&mut new_censoring, &self.censoring) {
                c[i] = old_c[idx];
            }
        }

        Ok(Self {
            design: DesignMatrix::Dense(new_design),
            response: new_response,
            weights: new_weights,
            censoring: new_censoring,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn dataset_validates_dimensions() {
        let x = Array2::<f64>::zeros((3, 2));
        let y = array![1.0, 2.0];
        let err = Dataset::new(x.clone(), y, None, None).unwrap_err();
        assert!(matches!(err, BoostlssError::DataError(_)));
    }

    #[test]
    fn dataset_rejects_negative_weights() {
        let x = Array2::<f64>::zeros((2, 2));
        let y = array![1.0, 2.0];
        let w = array![1.0, -0.5];
        let err = Dataset::new(x, y, Some(w), None).unwrap_err();
        assert!(matches!(err, BoostlssError::DataError(_)));
    }

    #[test]
    fn dataset_accepts_valid_data() {
        let x = Array2::<f64>::zeros((2, 2));
        let y = array![1.0, 2.0];
        let w = array![1.0, 1.0];
        let ds = Dataset::new(x, y, Some(w), None).unwrap();
        assert_eq!(ds.n_obs(), 2);
    }

    #[test]
    fn test_design_matrix_dense() {
        let dense = Array2::from_elem((3, 2), 1.0);
        let dm = DesignMatrix::Dense(dense);
        let col = dm.get_column(1).unwrap();
        assert_eq!(col, ndarray::Array1::from_elem(3, 1.0));
    }

    #[test]
    fn test_design_matrix_csc() {
        // [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
        let data = array![1.0, 3.0, 2.0, 4.0];
        let indices = array![0, 2, 1, 2];
        let indptr = array![0, 2, 4];
        let sparse = SparseMatrix::new(data, indices, indptr, (3, 2)).unwrap();
        let dm = DesignMatrix::Csc(sparse);

        let col0 = dm.get_column(0).unwrap();
        assert_eq!(col0, array![1.0, 0.0, 3.0]);

        let col1 = dm.get_column(1).unwrap();
        assert_eq!(col1, array![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_design_matrix_csr() {
        // [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]
        let data = array![1.0, 2.0, 3.0, 4.0];
        let indices = array![0, 1, 0, 1];
        let indptr = array![0, 1, 2, 4];
        let sparse = SparseMatrix::new(data, indices, indptr, (3, 2)).unwrap();
        let dm = DesignMatrix::Csr(sparse);

        let col0 = dm.get_column(0).unwrap();
        assert_eq!(col0, array![1.0, 0.0, 3.0]);

        let col1 = dm.get_column(1).unwrap();
        assert_eq!(col1, array![0.0, 2.0, 4.0]);
    }

    #[test]
    fn dataset_handles_censoring() {
        let x = Array2::<f64>::zeros((3, 2));
        let y = ndarray::array![1.0, 2.0, 3.0];
        let cens = ndarray::array![true, false, true];
        let ds = Dataset::new(x, y, None, Some(cens)).unwrap();
        assert_eq!(ds.censoring().unwrap().len(), 3);
    }

    #[test]
    fn dataset_rejects_invalid_censoring_length() {
        let x = Array2::<f64>::zeros((2, 2));
        let y = ndarray::array![1.0, 2.0];
        let cens = ndarray::array![true, false, true];
        let err = Dataset::new(x, y, None, Some(cens)).unwrap_err();
        assert!(matches!(err, BoostlssError::DataError(_)));
    }
}
