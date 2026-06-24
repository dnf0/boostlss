use crate::error::BoostlssError;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

const MAX_SAFE_COLS: usize = 100_000;
const MAX_SAFE_ELEMENTS: usize = 100_000_000;

/// Note: Currently, this relies on a dense matrix for the design matrix, which
/// means memory usage scales as O(n_obs * n_categories). For datasets with thousands
/// of categories, this guarantees massive wasted memory and limits the scale of data
/// that can be processed. A switch to sparse matrices (e.g., using `sprs`) should be
/// considered for a future refactor.
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

    pub fn build_design(&self, data: &crate::data::Dataset) -> Result<Array2<f64>, BoostlssError> {
        let col = data.design().get_column(self.feature_idx)?;
        let n_obs = col.len();
        if n_obs == 0 {
            return Ok(Array2::zeros((0, 0)));
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

        if max_idx >= MAX_SAFE_COLS {
            return Err(BoostlssError::DataError(
                format!("RandomEffects max_idx {} exceeds safe threshold ({}) and would cause excessive memory allocation", max_idx, MAX_SAFE_COLS)
            ));
        }

        let n_cols = max_idx + 1;

        let total_elements = n_obs.checked_mul(n_cols).ok_or_else(|| {
            BoostlssError::DataError("Dimensions overflow safe capacity".to_string())
        })?;

        if total_elements > MAX_SAFE_ELEMENTS {
            return Err(BoostlssError::DataError(
                format!("RandomEffects total elements {} exceeds safe threshold ({}) and would cause an OOM", total_elements, MAX_SAFE_ELEMENTS)
            ));
        }

        let mut design = Array2::zeros((n_obs, n_cols));

        for (i, &val) in col.iter().enumerate() {
            let idx = val as usize;
            design[[i, idx]] = 1.0;
        }

        Ok(design)
    }

    pub fn penalty_matrix(&self, n_cols: usize) -> Array2<f64> {
        Array2::eye(n_cols)
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
        let data = crate::data::Dataset::new(x, y, None).unwrap();
        let design = re.build_design(&data).unwrap();

        assert_eq!(design.shape(), &[4, 3]);
        assert_eq!(design.row(0), array![1.0, 0.0, 0.0].view());
        assert_eq!(design.row(1), array![0.0, 0.0, 1.0].view());
        assert_eq!(design.row(2), array![0.0, 1.0, 0.0].view());
        assert_eq!(design.row(3), array![1.0, 0.0, 0.0].view());
    }

    #[test]
    fn test_random_effects_invalid_data() {
        let re = RandomEffects::new(0);
        let x1 = array![[-1.0], [0.0]];
        let y1 = array![0.0, 0.0];
        let data1 = crate::data::Dataset::new(x1, y1, None).unwrap();
        assert!(re.build_design(&data1).is_err());

        let x2 = array![[0.5], [1.0]];
        let y2 = array![0.0, 0.0];
        let data2 = crate::data::Dataset::new(x2, y2, None).unwrap();
        assert!(re.build_design(&data2).is_err());
    }

    #[test]
    fn test_random_effects_oom_prevention() {
        let re = RandomEffects::new(0);

        let x1 = array![[1_000_000.0], [0.0]];
        let y1 = array![0.0, 0.0];
        let data1 = crate::data::Dataset::new(x1, y1, None).unwrap();
        let result = re.build_design(&data1);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("exceeds safe threshold"));
        } else {
            panic!("Expected an error for large index");
        }

        let x2 = ndarray::Array2::from_elem((2000, 1), 59_999.0);
        let y2 = ndarray::Array1::zeros(2000);
        let data2 = crate::data::Dataset::new(x2, y2, None).unwrap();
        let result2 = re.build_design(&data2);
        assert!(result2.is_err());
        if let Err(e) = result2 {
            assert!(e.to_string().contains("total elements"));
        } else {
            panic!("Expected an error for large total elements");
        }
    }

    #[test]
    fn test_random_effects_penalty() {
        let re = RandomEffects::new(0);
        let pen = re.penalty_matrix(3);
        assert_eq!(pen.shape(), &[3, 3]);
        assert_eq!(pen.diag(), array![1.0, 1.0, 1.0].view());
    }
}
