use crate::error::BoostlssError;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub design: Array2<f64>,
    pub response: Array1<f64>,
    pub weights: Option<Array1<f64>>,
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
