use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10;

fn default_gev_params() -> Vec<ParamSpec> {
    GEVLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEVLss {
    #[serde(skip, default = "default_gev_params")]
    params: Vec<ParamSpec>,
}

impl GEVLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", IdentityLink), // shape
            ],
        }
    }
}

impl Default for GEVLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for GEVLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;

        let mut nll = 0.0;
        let w = data.weights();
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(EPSILON);
            let nu = nu_link.response(eta[2][i]);
            let wi = w.map_or(1.0, |w_arr| w_arr[i]);

            let z = (yi - mu) / sigma;

            if nu.abs() < EPSILON {
                // Gumbel case
                let log_pdf = -sigma.ln() - z - (-z).exp();
                nll -= wi * log_pdf;
            } else {
                let term = 1.0 + nu * z;
                if term <= 0.0 {
                    return Err(BoostlssError::DataError(
                        "GEV support constraint violated".into(),
                    ));
                }
                let log_pdf = -sigma.ln() - (1.0 + 1.0 / nu) * term.ln() - term.powf(-1.0 / nu);
                nll -= wi * log_pdf;
            }
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
        let nu_link = &self.params[2].link;
        let mut grad = Array1::zeros(y.len());
        let w = data.weights();

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(EPSILON);
            let nu = nu_link.response(eta[2][i]);
            let wi = w.map_or(1.0, |w_arr| w_arr[i]);

            let z = (yi - mu) / sigma;

            let d_l_d_theta = if nu.abs() < EPSILON {
                // Gumbel case
                match k {
                    0 => {
                        // dL/dmu
                        (1.0 - (-z).exp()) / sigma
                    }
                    1 => {
                        // dL/dsigma
                        (-1.0 + z - z * (-z).exp()) / sigma
                    }
                    2 => {
                        // dL/dnu
                        0.0
                    }
                    _ => unreachable!(),
                }
            } else {
                let term = 1.0 + nu * z;
                if term <= 0.0 {
                    0.0
                } else {
                    let d_l_d_t =
                        -(1.0 + 1.0 / nu) / term + (1.0 / nu) * term.powf(-1.0 / nu - 1.0);
                    match k {
                        0 => {
                            // dL/dmu
                            d_l_d_t * (-nu / sigma)
                        }
                        1 => {
                            // dL/dsigma
                            d_l_d_t * (-nu * z / sigma) - 1.0 / sigma
                        }
                        2 => {
                            // dL/dnu
                            d_l_d_t * z
                                + (1.0 / (nu * nu)) * term.ln() * (1.0 - term.powf(-1.0 / nu))
                        }
                        _ => unreachable!(),
                    }
                }
            };

            let deriv = match k {
                0 => mu_link.deriv(mu),
                1 => sigma_link.deriv(sigma),
                2 => nu_link.deriv(nu),
                _ => unreachable!(),
            };

            grad[i] = wi * d_l_d_theta / deriv;
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let y_arr = y.clone();
        let w_arr = data.weights().cloned();

        let mut mu_val = 0.0;
        let mut sigma_val = 1.0;
        let mut nu_val = 0.1;

        let dense_design = match data.design() {
            crate::data::DesignMatrix::Dense(mat) => mat.clone(),
            _ => {
                return Err(BoostlssError::DataError(
                    "gev requires dense matrix".to_string(),
                ))
            }
        };
        let ds = Dataset::new(dense_design, y_arr.clone(), w_arr.clone()).unwrap();

        let mut eta = vec![
            Array1::from_elem(y_arr.len(), self.params[0].link.link(mu_val)),
            Array1::from_elem(y_arr.len(), self.params[1].link.link(sigma_val)),
            Array1::from_elem(y_arr.len(), self.params[2].link.link(nu_val)),
        ];

        for _ in 0..2 {
            let opt_eta_mu = minimize_1d(
                |m| {
                    eta[0].fill(m);
                    eta[1].fill(self.params[1].link.link(sigma_val));
                    eta[2].fill(self.params[2].link.link(nu_val));
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                -10.0,
                10.0,
            );
            mu_val = self.params[0].link.response(opt_eta_mu);

            let opt_eta_sigma = minimize_1d(
                |s| {
                    eta[0].fill(self.params[0].link.link(mu_val));
                    eta[1].fill(s);
                    eta[2].fill(self.params[2].link.link(nu_val));
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            sigma_val = self.params[1].link.response(opt_eta_sigma);

            let opt_eta_nu = minimize_1d(
                |n| {
                    eta[0].fill(self.params[0].link.link(mu_val));
                    eta[1].fill(self.params[1].link.link(sigma_val));
                    eta[2].fill(n);
                    self.nll(&ds, &eta).unwrap_or(f64::MAX)
                },
                -1.0,
                1.0,
            );
            nu_val = self.params[2].link.response(opt_eta_nu);
        }

        Ok(vec![
            self.params[0].link.link(mu_val),
            self.params[1].link.link(sigma_val),
            self.params[2].link.link(nu_val),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_gev_init() {
        let fam = GEVLss::new();
        let y = array![1.0, 2.0, 3.0];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
        assert!(fam.init_offsets(&ds).is_ok());
    }

    #[test]
    fn test_gev_gradients() {
        let fam = GEVLss::new();
        let y = array![1.0, 2.0, 3.0];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
        let eta = vec![
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0], // log(sigma) = 0 => sigma = 1
            array![0.1, 0.1, 0.1], // general case
        ];

        let grad_mu = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_sigma = fam.ngradient(&ds, &eta, 1).unwrap();
        let grad_nu = fam.ngradient(&ds, &eta, 2).unwrap();

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
            assert!(
                approx::relative_eq!(grad_sigma[i], fin_diff, epsilon = 1e-4, max_relative = 1e-3),
                "grad_sigma[{}] = {}, fin_diff = {}",
                i,
                grad_sigma[i],
                fin_diff
            );
            eta_plus[1][i] -= eps;
            eta_minus[1][i] += eps;
        }

        for i in 0..3 {
            eta_plus[2][i] += eps;
            eta_minus[2][i] -= eps;
            let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap() - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_nu[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
            eta_plus[2][i] -= eps;
            eta_minus[2][i] += eps;
        }
    }

    #[test]
    fn test_gev_gumbel_gradients() {
        let fam = GEVLss::new();
        let y = array![1.0, 2.0, 3.0];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
        let eta = vec![
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0], // gumbel case
        ];

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
            assert!(
                approx::relative_eq!(grad_sigma[i], fin_diff, epsilon = 1e-4, max_relative = 1e-3),
                "grad_sigma[{}] = {}, fin_diff = {}",
                i,
                grad_sigma[i],
                fin_diff
            );
            eta_plus[1][i] -= eps;
            eta_minus[1][i] += eps;
        }
    }
}
