use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_gpd_params() -> Vec<ParamSpec> {
    GpdLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpdLss {
    #[serde(skip, default = "default_gpd_params")]
    params: Vec<ParamSpec>,
}

impl GpdLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("xi", IdentityLink),
            ],
        }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val < 0.0) {
            return Err(BoostlssError::DataError(
                "GPD response must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for GpdLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for GpdLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let w = data.weights();
        let sigma_link = &self.params[0].link;
        let xi_link = &self.params[1].link;

        let mut total_nll = 0.0;

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let sigma = sigma_link.response(eta[0][i]).max(1e-10);
            let xi = xi_link.response(eta[1][i]);

            let z = xi * yi / sigma;

            // Check support: 1 + xi * y / sigma > 0
            if 1.0 + z <= 0.0 {
                total_nll += wi * 1e10; // Large penalty
                continue;
            }

            let nll = if xi.abs() < 1e-6 {
                sigma.ln() + yi / sigma
            } else {
                sigma.ln() + (1.0 / xi + 1.0) * (1.0 + z).ln()
            };

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

        let mut sigma_val = y.mean().unwrap_or(1.0).max(1e-3);
        let mut xi_val = 0.1;

        for _ in 0..3 {
            let eta_xi = xi_val;
            let opt_sigma_ln = minimize_1d(
                |s_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), s_ln),
                        Array1::from_elem(y.len(), eta_xi),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            sigma_val = opt_sigma_ln.exp();

            let eta_sigma = sigma_val.ln();
            let opt_xi = minimize_1d(
                |x| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_sigma),
                        Array1::from_elem(y.len(), x),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -2.0,
                2.0,
            );
            xi_val = opt_xi;
        }

        Ok(vec![
            self.params[0].link.link(sigma_val),
            self.params[1].link.link(xi_val),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_gpd_gradients() {
        let fam = GpdLss::new();
        let y = array![0.5, 1.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None, None).unwrap();
        let eta = vec![array![0.0, 0.5, 1.0, 1.5], array![0.1, 0.2, 0.3, 0.4]];

        let grad_sigma = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_xi = fam.ngradient(&ds, &eta, 1).unwrap();
        let eps = 1e-5;

        for i in 0..4 {
            let mut eta_plus = eta.clone();
            let mut eta_minus = eta.clone();

            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;
            let fin_diff_sigma = -(fam.nll(&ds, &eta_plus).unwrap()
                - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_sigma[i],
                fin_diff_sigma,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));

            eta_plus[0][i] -= eps;
            eta_minus[0][i] += eps;

            eta_plus[1][i] += eps;
            eta_minus[1][i] -= eps;
            let fin_diff_xi = -(fam.nll(&ds, &eta_plus).unwrap()
                - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_xi[i],
                fin_diff_xi,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
        }
    }
}
