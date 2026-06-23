use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, LogitLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10;

fn default_zip_params() -> Vec<ParamSpec> {
    ZIPLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZIPLss {
    #[serde(skip, default = "default_zip_params")]
    params: Vec<ParamSpec>,
}

impl ZIPLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogLink),
                ParamSpec::new("sigma", LogitLink), // Zero inflation probability
            ],
        }
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val < 0.0 || val.fract() != 0.0) {
            return Err(BoostlssError::DataError(
                "ZIP response must be non-negative integers".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for ZIPLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for ZIPLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;

        let mut nll = 0.0;
        let w = data.weights();
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(EPSILON);
            let sigma = sigma_link.response(eta[1][i]).clamp(EPSILON, 1.0 - EPSILON);
            let wi = w.map_or(1.0, |w| w[i]);

            let log_pdf = if yi == 0.0 {
                (sigma + (1.0 - sigma) * (-mu).exp()).ln()
            } else {
                (1.0 - sigma).ln() + yi * mu.ln() - mu - statrs::function::gamma::ln_gamma(yi + 1.0)
            };
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
        let w = data.weights();

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(EPSILON);
            let sigma = sigma_link.response(eta[1][i]).clamp(EPSILON, 1.0 - EPSILON);
            let wi = w.map_or(1.0, |weights| weights[i]);

            if k == 0 {
                let d_l_d_mu = if yi == 0.0 {
                    let num = -(1.0 - sigma) * (-mu).exp();
                    let den = sigma + (1.0 - sigma) * (-mu).exp();
                    num / den
                } else {
                    (yi / mu) - 1.0
                };
                let d_mu_d_eta = 1.0 / mu_link.deriv(mu);
                grad[i] = wi * d_l_d_mu * d_mu_d_eta;
            } else {
                let d_l_d_sigma = if yi == 0.0 {
                    (1.0 - (-mu).exp()) / (sigma + (1.0 - sigma) * (-mu).exp())
                } else {
                    -1.0 / (1.0 - sigma)
                };
                let d_sigma_d_eta = 1.0 / sigma_link.deriv(sigma);
                grad[i] = wi * d_l_d_sigma * d_sigma_d_eta;
            }
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y_arr = data.response().clone();
        let y_len = y_arr.len();

        let dummy_ds = Dataset::new(
            ndarray::Array2::<f64>::zeros((y_len, 1)),
            y_arr,
            data.weights().cloned(),
        )?;

        let mut mu_val = 1.0;
        let mut sigma_val = 0.5;

        for _ in 0..3 {
            let log_mu = minimize_1d(
                |m| {
                    let eta = vec![
                        Array1::from_elem(y_len, m),
                        Array1::from_elem(y_len, self.params[1].link.link(sigma_val)),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            mu_val = self.params[0].link.response(log_mu);

            let logit_sigma = minimize_1d(
                |s| {
                    let eta = vec![
                        Array1::from_elem(y_len, self.params[0].link.link(mu_val)),
                        Array1::from_elem(y_len, s),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            sigma_val = self.params[1].link.response(logit_sigma);
        }

        Ok(vec![
            self.params[0].link.link(mu_val),
            self.params[1].link.link(sigma_val),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_zip_gradients() {
        let fam = ZIPLss::new();
        let y = array![0.0, 0.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None).unwrap();
        let eta = vec![array![0.0, 0.0, 1.0, 2.0], array![-1.0, 0.0, 1.0, 2.0]];

        let grad_mu = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_sigma = fam.ngradient(&ds, &eta, 1).unwrap();

        let eps = 1e-5;
        let mut eta_plus = eta.clone();
        let mut eta_minus = eta.clone();

        for i in 0..4 {
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

        for i in 0..4 {
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
