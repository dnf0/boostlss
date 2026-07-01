use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

fn default_log_normal_params() -> Vec<ParamSpec> {
    LogNormalLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogNormalLss {
    #[serde(skip, default = "default_log_normal_params")]
    params: Vec<ParamSpec>,
}

impl LogNormalLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
            ],
        }
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val <= 0.0) {
            return Err(BoostlssError::DataError(
                "Log-Normal response must be strictly positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for LogNormalLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for LogNormalLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let weights = data.weights();

        let mut nll = 0.0;

        for i in 0..y.len() {
            let yi = y[i];
            let mui = eta[0][i];
            let sigmai = eta[1][i];
            let wi = weights.map_or(1.0, |w| w[i]);

            let mu = mu_link.response(mui);
            let sigma = sigma_link.response(sigmai).max(EPSILON);

            let log_y = yi.ln();
            let log_pdf =
                -log_y - sigma.ln() - 0.5 * (2.0 * PI).ln() - 0.5 * ((log_y - mu) / sigma).powi(2);
            nll -= wi * log_pdf;
        }

        Ok(nll)
    }

    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let weights = data.weights();
        let mut grad = Array1::zeros(y.len());

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(EPSILON);
            let wi = weights.map_or(1.0, |w| w[i]);
            let log_y = yi.ln();

            if k == 0 {
                let d_l_d_mu = (log_y - mu) / (sigma * sigma);
                let d_mu_d_eta = 1.0 / mu_link.deriv(mu);
                grad[i] = wi * d_l_d_mu * d_mu_d_eta;
            } else {
                let d_l_d_sigma = ((log_y - mu).powi(2) / (sigma * sigma * sigma)) - (1.0 / sigma);
                let d_sigma_d_eta = 1.0 / sigma_link.deriv(sigma);
                grad[i] = wi * d_l_d_sigma * d_sigma_d_eta;
            }
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;
        let log_y = data.response().mapv(|val| val.ln());
        let mu = weighted_mean(&log_y, data.weights());
        let sigma = weighted_sd(&log_y, data.weights()).max(EPSILON);

        Ok(vec![
            self.params[0].link.link(mu),
            self.params[1].link.link(sigma),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_log_normal_invalid_response() {
        let fam = LogNormalLss::new();

        let y_zero = array![1.0, 0.0, 3.0];
        let ds_zero = Dataset::new(Array2::<f64>::zeros((3, 1)), y_zero, None, None).unwrap();
        assert!(matches!(
            fam.init_offsets(&ds_zero),
            Err(BoostlssError::DataError(_))
        ));

        let y_neg = array![1.0, -1.0, 3.0];
        let ds_neg = Dataset::new(Array2::<f64>::zeros((3, 1)), y_neg, None, None).unwrap();
        assert!(matches!(
            fam.init_offsets(&ds_neg),
            Err(BoostlssError::DataError(_))
        ));
    }

    #[test]
    fn test_log_normal_gradients() {
        let fam = LogNormalLss::new();
        let y = array![1.0, std::f64::consts::E, 7.389];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None, None).unwrap();
        let eta = vec![array![0.0, 1.0, 2.0], array![-0.5, 0.0, 0.5]];

        let grad_mu = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_sigma = fam.ngradient(&ds, &eta, 1).unwrap();

        let eps = 1e-5;
        let mut eta_plus = eta.clone();
        let mut eta_minus = eta.clone();

        for i in 0..3 {
            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;
            let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap() - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_mu[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
            eta_plus[0][i] -= eps;
            eta_minus[0][i] += eps;
        }

        for i in 0..3 {
            eta_plus[1][i] += eps;
            eta_minus[1][i] -= eps;
            let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap() - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_sigma[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
            eta_plus[1][i] -= eps;
            eta_minus[1][i] += eps;
        }
    }
}
