use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10;

fn default_weibull_params() -> Vec<ParamSpec> {
    WeibullLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeibullLss {
    #[serde(skip, default = "default_weibull_params")]
    params: Vec<ParamSpec>,
}

impl WeibullLss {
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
                "Weibull response must be strictly positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for WeibullLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for WeibullLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;

        let mut nll = 0.0;

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(EPSILON);
            let sigma = sigma_link.response(eta[1][i]).max(EPSILON);
            let wi = data.weights().map_or(1.0, |w| w[i]);

            // log_pdf = log(sigma) - log(mu) + (sigma - 1)*(log(y) - log(mu)) - (y/mu)^sigma
            let log_pdf =
                sigma.ln() - mu.ln() + (sigma - 1.0) * (yi.ln() - mu.ln()) - (yi / mu).powf(sigma);
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
        let mut grad = Array1::zeros(y.len());

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(EPSILON);
            let sigma = sigma_link.response(eta[1][i]).max(EPSILON);
            let wi = data.weights().map_or(1.0, |w| w[i]);

            if k == 0 {
                let d_l_d_mu = (sigma / mu) * ((yi / mu).powf(sigma) - 1.0);
                let d_mu_d_eta = 1.0 / mu_link.deriv(mu);
                grad[i] = wi * d_l_d_mu * d_mu_d_eta;
            } else {
                let y_over_mu = yi / mu;
                let d_l_d_sigma = (1.0 / sigma) + y_over_mu.ln() * (1.0 - y_over_mu.powf(sigma));
                let d_sigma_d_eta = 1.0 / sigma_link.deriv(sigma);
                grad[i] = wi * d_l_d_sigma * d_sigma_d_eta;
            }
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        // Optimize both via 1D line searches iteratively
        let y = data.response();

        let dummy_ds = Dataset::new(
            ndarray::Array2::zeros((y.len(), 0)),
            y.clone(),
            data.weights().cloned(),
            data.censoring().cloned(),
        )
        .unwrap();

        let mut mu_val: f64 = 1.0;
        let mut sigma_val: f64 = 1.0;

        for _ in 0..3 {
            // Few iterations of coordinate descent
            let log_mu = minimize_1d(
                |m| {
                    let eta = vec![
                        Array1::from_elem(y.len(), m),
                        Array1::from_elem(y.len(), sigma_val.ln()),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -10.0,
                10.0,
            );
            mu_val = log_mu.exp();

            let log_sigma = minimize_1d(
                |s| {
                    let eta = vec![
                        Array1::from_elem(y.len(), mu_val.ln()),
                        Array1::from_elem(y.len(), s),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            sigma_val = log_sigma.exp();
        }

        Ok(vec![mu_val.ln(), sigma_val.ln()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_weibull_gradients() {
        let fam = WeibullLss::new();
        let y = array![1.5, 2.5, 5.0];
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
