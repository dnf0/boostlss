use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

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
}
