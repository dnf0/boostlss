use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_ig_params() -> Vec<ParamSpec> {
    InverseGaussianLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InverseGaussianLss {
    #[serde(skip, default = "default_ig_params")]
    params: Vec<ParamSpec>,
}

impl InverseGaussianLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogLink),
                ParamSpec::new("sigma", LogLink),
            ],
        }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val <= 0.0) {
            return Err(BoostlssError::DataError(
                "Inverse Gaussian response must be strictly positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for InverseGaussianLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for InverseGaussianLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let w = data.weights();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;

        let mut total_nll = 0.0;
        let pi = std::f64::consts::PI;

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);

            // NLL = 0.5 * ln(2 * pi * sigma^2 * y^3) + (y - mu)^2 / (2 * sigma^2 * mu^2 * y)
            let log_term = 0.5 * (2.0 * pi * sigma * sigma * yi * yi * yi).ln();
            let exp_term = (yi - mu) * (yi - mu) / (2.0 * sigma * sigma * mu * mu * yi);

            total_nll += wi * (log_term + exp_term);
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();

        let mu_init = y.mean().unwrap_or(1.0).max(1e-5);
        let mut sum_diff = 0.0;
        for &yi in y {
            sum_diff += (1.0 / yi) - (1.0 / mu_init);
        }
        let sigma_sq_init = (sum_diff / y.len() as f64).max(1e-5);
        let sigma_init = sigma_sq_init.sqrt();

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_ig_gradients() {
        let fam = InverseGaussianLss::new();
        let y = array![0.5, 1.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None, None).unwrap();
        let eta = vec![array![-1.0, 0.0, 1.0, 2.0], array![-0.5, 0.0, 0.5, 1.0]];

        let grad_mu = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_sigma = fam.ngradient(&ds, &eta, 1).unwrap();
        let eps = 1e-5;

        for i in 0..4 {
            let mut eta_plus = eta.clone();
            let mut eta_minus = eta.clone();

            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;
            let fin_diff_mu = -(fam.nll(&ds, &eta_plus).unwrap()
                - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_mu[i],
                fin_diff_mu,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));

            eta_plus[0][i] -= eps;
            eta_minus[0][i] += eps;

            eta_plus[1][i] += eps;
            eta_minus[1][i] -= eps;
            let fin_diff_sigma = -(fam.nll(&ds, &eta_plus).unwrap()
                - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_sigma[i],
                fin_diff_sigma,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
        }
    }
}
