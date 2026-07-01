use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_burr12_params() -> Vec<ParamSpec> {
    Burr12Lss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Burr12Lss {
    #[serde(skip, default = "default_burr12_params")]
    params: Vec<ParamSpec>,
}

impl Burr12Lss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogLink),    // scale
                ParamSpec::new("sigma", LogLink), // shape1 (c)
                ParamSpec::new("nu", LogLink),    // shape2 (k)
            ],
        }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val <= 0.0) {
            return Err(BoostlssError::DataError(
                "Burr12 response must be strictly positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for Burr12Lss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for Burr12Lss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let w = data.weights();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;

        let mut total_nll = 0.0;

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]).max(1e-10);

            let z = (yi / mu).powf(sigma);

            // NLL = -ln(sigma) - ln(nu) - (sigma - 1)*ln(y) + sigma*ln(mu) + (nu + 1)*ln(1 + (y/mu)^sigma)
            let nll = -sigma.ln() - nu.ln() - (sigma - 1.0) * yi.ln()
                + sigma * mu.ln()
                + (nu + 1.0) * (1.0 + z).ln();

            total_nll += wi * nll;
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let dummy_ds = Dataset::new(
            ndarray::Array2::zeros((y.len(), 0)),
            y.clone(),
            data.weights().cloned(),
            data.censoring().cloned(),
        )
        .unwrap();

        let mut mu_val = y.mean().unwrap_or(1.0).max(1e-3);
        let mut sigma_val: f64 = 1.0;
        let mut nu_val: f64 = 1.0;

        for _ in 0..3 {
            let eta_sigma = sigma_val.ln();
            let eta_nu = nu_val.ln();

            let opt_mu_ln = minimize_1d(
                |m_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), m_ln),
                        Array1::from_elem(y.len(), eta_sigma),
                        Array1::from_elem(y.len(), eta_nu),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            mu_val = opt_mu_ln.exp();

            let eta_mu = mu_val.ln();
            let opt_sigma_ln = minimize_1d(
                |s_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_mu),
                        Array1::from_elem(y.len(), s_ln),
                        Array1::from_elem(y.len(), eta_nu),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            sigma_val = opt_sigma_ln.exp();

            let eta_sigma = sigma_val.ln();
            let opt_nu_ln = minimize_1d(
                |n_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_mu),
                        Array1::from_elem(y.len(), eta_sigma),
                        Array1::from_elem(y.len(), n_ln),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            nu_val = opt_nu_ln.exp();
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
    fn test_burr12_gradients() {
        let fam = Burr12Lss::new();
        let y = array![0.5, 1.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None, None).unwrap();
        let eta = vec![
            array![0.0, 0.5, 1.0, 1.5],  // ln(mu)
            array![0.1, 0.2, 0.3, 0.4],  // ln(sigma)
            array![0.0, 0.1, -0.1, 0.5], // ln(nu)
        ];

        let eps = 1e-5;

        for k in 0..3 {
            let grad = fam.ngradient(&ds, &eta, k).unwrap();
            for i in 0..4 {
                let mut eta_plus = eta.clone();
                let mut eta_minus = eta.clone();

                eta_plus[k][i] += eps;
                eta_minus[k][i] -= eps;
                let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap()
                    - fam.nll(&ds, &eta_minus).unwrap())
                    / (2.0 * eps);
                assert!(approx::relative_eq!(
                    grad[i],
                    fin_diff,
                    epsilon = 1e-4,
                    max_relative = 1e-3
                ));
            }
        }
    }
}
