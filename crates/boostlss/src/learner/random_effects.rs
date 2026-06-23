use crate::error::BoostlssError;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffects {
    pub feature: String,
    pub df: f64,
}

impl RandomEffects {
    pub fn new(feature: &str) -> Self {
        Self {
            feature: feature.to_string(),
            df: 4.0,
        }
    }

    pub fn df(mut self, df: f64) -> Self {
        self.df = df;
        self
    }

    pub fn build_design(&self, x: &Array1<f64>) -> Result<Array2<f64>, BoostlssError> {
        let n_obs = x.len();
        if n_obs == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let mut max_idx = 0;
        for &val in x.iter() {
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

        if n_cols > 100_000 {
            return Err(BoostlssError::DataError(
                format!("RandomEffects max_idx {} exceeds safe threshold (100_000) and would cause excessive memory allocation", max_idx)
            ));
        }

        let mut design = Array2::zeros((n_obs, n_cols));

        for (i, &val) in x.iter().enumerate() {
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
        let re = RandomEffects::new("group");
        let x = array![0.0, 2.0, 1.0, 0.0];
        let design = re.build_design(&x).unwrap();

        assert_eq!(design.shape(), &[4, 3]);
        assert_eq!(design.row(0), array![1.0, 0.0, 0.0].view());
        assert_eq!(design.row(1), array![0.0, 0.0, 1.0].view());
        assert_eq!(design.row(2), array![0.0, 1.0, 0.0].view());
        assert_eq!(design.row(3), array![1.0, 0.0, 0.0].view());
    }

    #[test]
    fn test_random_effects_invalid_data() {
        let re = RandomEffects::new("group");
        assert!(re.build_design(&array![-1.0, 0.0]).is_err());
        assert!(re.build_design(&array![0.5, 1.0]).is_err());
    }

    #[test]
    fn test_random_effects_oom_prevention() {
        let re = RandomEffects::new("group");
        let result = re.build_design(&array![1_000_000.0, 0.0]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("exceeds safe threshold"));
        } else {
            panic!("Expected an error for large index");
        }
    }

    #[test]
    fn test_random_effects_penalty() {
        let re = RandomEffects::new("group");
        let pen = re.penalty_matrix(3);
        assert_eq!(pen.shape(), &[3, 3]);
        assert_eq!(pen.diag(), array![1.0, 1.0, 1.0].view());
    }
}
