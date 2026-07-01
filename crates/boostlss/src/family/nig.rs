use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::minimize_1d;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_nig_params() -> Vec<ParamSpec> {
    NigLss::new().params
}

/// Normal Inverse Gaussian (NIG) distribution.
/// Parameterized by:
/// 1. mu (location) -> IdentityLink
/// 2. delta (scale) -> LogLink
/// 3. beta (skewness) -> IdentityLink
/// 4. gamma (tail shape) -> LogLink
///
/// Note: The standard alpha parameter is derived as alpha = sqrt(gamma^2 + beta^2),
/// which automatically enforces the constraint 0 <= |beta| < alpha.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NigLss {
    #[serde(skip, default = "default_nig_params")]
    params: Vec<ParamSpec>,
}

impl NigLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("delta", LogLink),
                ParamSpec::new("beta", IdentityLink),
                ParamSpec::new("gamma", LogLink),
            ],
        }
    }

    /// Robust log of the modified Bessel function of the second kind K_1(z).
    /// For large z, it uses the asymptotic expansion to avoid underflow.
    fn ln_k1(x: f64) -> f64 {
        if x > 50.0 {
            0.5 * (std::f64::consts::PI / (2.0 * x)).ln() - x
        } else if x > 2.0 {
            let inv_x = 1.0 / x;
            let p = 1.25331414
                + inv_x
                    * (0.23498619
                        + inv_x
                            * (-0.03655620
                                + inv_x
                                    * (0.01504268
                                        + inv_x
                                            * (-0.00780353
                                                + inv_x * (0.00325614 + inv_x * (-0.00068245))))));
            (p / x.sqrt()).ln() - x
        } else {
            let t = x / 2.0;
            let t2 = t * t;
            let p1 = 1.0
                + t2 * (0.15443144
                    + t2 * (-0.67278579
                        + t2 * (-0.18156897
                            + t2 * (-0.01919402 + t2 * (-0.00110404 + t2 * -0.00004686)))));

            let t_i = x / 3.75;
            let t_i2 = t_i * t_i;
            let i1_div_x = 0.5
                + t_i2
                    * (0.87890594
                        + t_i2
                            * (0.51498869
                                + t_i2
                                    * (0.15084934
                                        + t_i2
                                            * (0.02658733
                                                + t_i2 * (0.00301532 + t_i2 * 0.00032411)))));
            let i1 = x * i1_div_x;

            let k1 = (p1 + x * x * t.ln() * i1) / x;
            k1.ln()
        }
    }
}

impl Default for NigLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for NigLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let w = data.weights();

        let mu_link = &self.params[0].link;
        let delta_link = &self.params[1].link;
        let beta_link = &self.params[2].link;
        let gamma_link = &self.params[3].link;

        let mut total_nll = 0.0;
        let pi = std::f64::consts::PI;

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);

            let mu = mu_link.response(eta[0][i]);
            let delta = delta_link.response(eta[1][i]).max(1e-10);
            let beta = beta_link.response(eta[2][i]);
            let gamma = gamma_link.response(eta[3][i]).max(1e-10);

            // alpha = sqrt(gamma^2 + beta^2)
            let alpha = (gamma * gamma + beta * beta).sqrt();

            let diff = yi - mu;
            let d_sq = delta * delta;
            let term = (d_sq + diff * diff).sqrt();
            let z = alpha * term;

            let log_pdf = alpha.ln() + delta.ln() + Self::ln_k1(z) - pi.ln() - term.ln()
                + delta * gamma
                + beta * diff;

            total_nll -= wi * log_pdf;
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
        let mut delta_val = y.var(1.0).sqrt().max(1e-3);
        let mut beta_val = 0.0;
        let mut gamma_val: f64 = 1.0;

        for _ in 0..3 {
            let eta_delta = delta_val.ln();
            let eta_beta = beta_val;
            let eta_gamma = gamma_val.ln();

            let opt_mu = minimize_1d(
                |m| {
                    let eta = vec![
                        Array1::from_elem(y.len(), m),
                        Array1::from_elem(y.len(), eta_delta),
                        Array1::from_elem(y.len(), eta_beta),
                        Array1::from_elem(y.len(), eta_gamma),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                mu_val - 5.0,
                mu_val + 5.0,
            );
            mu_val = opt_mu;

            let eta_mu = mu_val;
            let opt_delta_ln = minimize_1d(
                |d_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_mu),
                        Array1::from_elem(y.len(), d_ln),
                        Array1::from_elem(y.len(), eta_beta),
                        Array1::from_elem(y.len(), eta_gamma),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            delta_val = opt_delta_ln.exp();

            let eta_delta = delta_val.ln();
            let opt_beta = minimize_1d(
                |b| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_mu),
                        Array1::from_elem(y.len(), eta_delta),
                        Array1::from_elem(y.len(), b),
                        Array1::from_elem(y.len(), eta_gamma),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            beta_val = opt_beta;

            let eta_beta = beta_val;
            let opt_gamma_ln = minimize_1d(
                |g_ln| {
                    let eta = vec![
                        Array1::from_elem(y.len(), eta_mu),
                        Array1::from_elem(y.len(), eta_delta),
                        Array1::from_elem(y.len(), eta_beta),
                        Array1::from_elem(y.len(), g_ln),
                    ];
                    self.nll(&dummy_ds, &eta).unwrap_or(f64::MAX)
                },
                -5.0,
                5.0,
            );
            gamma_val = opt_gamma_ln.exp();
        }

        Ok(vec![
            self.params[0].link.link(mu_val),
            self.params[1].link.link(delta_val),
            self.params[2].link.link(beta_val),
            self.params[3].link.link(gamma_val),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_nig_init() {
        let fam = NigLss::new();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, None, None).unwrap();

        let offsets = fam.init_offsets(&ds).unwrap();
        assert_eq!(offsets.len(), 4);
    }

    #[test]
    fn test_nig_gradients() {
        let fam = NigLss::new();
        let y = array![1.5, -0.5, 3.0];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None, None).unwrap();
        let eta = vec![
            array![0.0, 1.0, -1.0], // mu
            array![-0.5, 0.0, 0.5], // ln(delta)
            array![0.1, -0.2, 0.3], // beta
            array![-0.1, 0.2, 0.0], // ln(gamma)
        ];

        let eps = 1e-5;

        for k in 0..4 {
            let grad = fam.ngradient(&ds, &eta, k).unwrap();
            let mut eta_plus = eta.clone();
            let mut eta_minus = eta.clone();

            for i in 0..3 {
                eta_plus[k][i] += eps;
                eta_minus[k][i] -= eps;
                let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap()
                    - fam.nll(&ds, &eta_minus).unwrap())
                    / (2.0 * eps);

                assert_relative_eq!(grad[i], fin_diff, epsilon = 1e-4, max_relative = 1e-3);

                eta_plus[k][i] -= eps;
                eta_minus[k][i] += eps;
            }
        }
    }
}
