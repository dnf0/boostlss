use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

fn default_ged_params() -> Vec<ParamSpec> {
    GedLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GedLss {
    #[serde(skip, default = "default_ged_params")]
    params: Vec<ParamSpec>,
}

impl GedLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", LogLink),
            ],
        }
    }
}

impl Default for GedLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for GedLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let w = data.weights();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;

        let mut total_nll = 0.0;

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]).max(1e-10);

            let abs_z = ((yi - mu) / sigma).abs();

            // NLL = ln(2 * sigma * Gamma(1/nu) / nu) + abs_z^nu
            //     = ln(2) + ln(sigma) + ln_gamma(1/nu) - ln(nu) + abs_z^nu
            let nll =
                std::f64::consts::LN_2 + sigma.ln() + ln_gamma(1.0 / nu) - nu.ln() + abs_z.powf(nu);

            total_nll += wi * nll;
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let dummy_ds = Dataset::new(
            ndarray::Array2::zeros((y.len(), 0)),
            y.clone(),
            data.weights().cloned(),
            data.censoring().cloned(),
        )
        .unwrap();

        let mut mu_val = y.mean().unwrap_or(0.0);
        let mut sigma_val = y.var(1.0).sqrt().max(1e-3);
        let mut nu_val: f64 = 2.0; // Normal distribution is nu=2

        for _ in 0..3 {
            let eta_sigma = sigma_val.ln();
            let eta_nu = nu_val.ln();

            let opt_mu = minimize_1d(
                |m| {
                    let eta = vec![
                        Array1::from_elem(y.len(), m),
                        Array1::from_elem(y.len(), eta_sigma),
                        Array1::from_elem(y.len(), eta_nu),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                mu_val - 5.0,
                mu_val + 5.0,
            );
            mu_val = opt_mu;

            let eta_mu = mu_val;
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
                -2.0,
                2.0,
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
    fn test_ged_gradients() {
        let fam = GedLss::new();
        let y = array![0.5, -1.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None, None).unwrap();
        let eta = vec![
            array![0.0, 0.5, 1.0, 1.5],  // mu
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
