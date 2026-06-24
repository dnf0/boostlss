use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub data: Array1<f64>,
    pub indices: Array1<usize>,
    pub indptr: Array1<usize>,
    pub shape: (usize, usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DesignMatrix {
    Dense(Array2<f64>),
    Csr(SparseMatrix),
    Csc(SparseMatrix),
}

impl DesignMatrix {
    pub fn get_column(&self, col_idx: usize) -> Result<Array1<f64>, BoostlssError> {
        match self {
            Self::Dense(mat) => {
                if col_idx >= mat.ncols() {
                    return Err(BoostlssError::DataError(
                        "Column index out of bounds".to_string(),
                    ));
                }
                Ok(mat.column(col_idx).to_owned())
            }
            Self::Csc(sparse) => {
                if col_idx >= sparse.shape.1 {
                    return Err(BoostlssError::DataError(
                        "Column index out of bounds".to_string(),
                    ));
                }
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
                if col_idx >= sparse.shape.1 {
                    return Err(BoostlssError::DataError(
                        "Column index out of bounds".to_string(),
                    ));
                }
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
    design: Array2<f64>,
    response: Array1<f64>,
    weights: Option<Array1<f64>>,
}

impl Dataset {
    pub fn new(
        design: Array2<f64>,
        response: Array1<f64>,
        weights: Option<Array1<f64>>,
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
        Ok(Self {
            design,
            response,
            weights,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.design.nrows()
    }

    pub fn design(&self) -> &Array2<f64> {
        &self.design
    }

    pub fn response(&self) -> &Array1<f64> {
        &self.response
    }

    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
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
        })
    }

    pub fn subset(&self, indices: &[usize]) -> Result<Self, BoostlssError> {
        let n = indices.len();
        let mut new_design = ndarray::Array2::zeros((n, self.design.ncols()));
        let mut new_response = ndarray::Array1::zeros(n);
        let mut new_weights = self.weights.as_ref().map(|_| ndarray::Array1::zeros(n));

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.design.nrows() {
                return Err(BoostlssError::DataError("Index out of bounds".to_string()));
            }
            new_design.row_mut(i).assign(&self.design.row(idx));
            new_response[i] = self.response[idx];
            if let Some(ref mut w) = new_weights {
                w[i] = self.weights.as_ref().unwrap()[idx];
            }
        }

        Ok(Self {
            design: new_design,
            response: new_response,
            weights: new_weights,
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
        let err = Dataset::new(x.clone(), y, None).unwrap_err();
        assert!(matches!(err, BoostlssError::DataError(_)));
    }

    #[test]
    fn dataset_rejects_negative_weights() {
        let x = Array2::<f64>::zeros((2, 2));
        let y = array![1.0, 2.0];
        let w = array![1.0, -0.5];
        let err = Dataset::new(x, y, Some(w)).unwrap_err();
        assert!(matches!(err, BoostlssError::DataError(_)));
    }

    #[test]
    fn dataset_accepts_valid_data() {
        let x = Array2::<f64>::zeros((2, 2));
        let y = array![1.0, 2.0];
        let w = array![1.0, 1.0];
        let ds = Dataset::new(x, y, Some(w)).unwrap();
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
        let sparse = SparseMatrix {
            data,
            indices,
            indptr,
            shape: (3, 2),
        };
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
        let sparse = SparseMatrix {
            data,
            indices,
            indptr,
            shape: (3, 2),
        };
        let dm = DesignMatrix::Csr(sparse);

        let col0 = dm.get_column(0).unwrap();
        assert_eq!(col0, array![1.0, 0.0, 3.0]);

        let col1 = dm.get_column(1).unwrap();
        assert_eq!(col1, array![0.0, 2.0, 4.0]);
    }
}
