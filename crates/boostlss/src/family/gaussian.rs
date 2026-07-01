use super::Family;
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

fn default_gaussian_params() -> Vec<ParamSpec> {
    GaussianLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianLss {
    #[serde(skip, default = "default_gaussian_params")]
    params: Vec<ParamSpec>,
}

impl GaussianLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
            ],
        }
    }
}

impl Default for GaussianLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for GaussianLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let mu = eta[0].mapv(|x| self.params[0].link.response(x));
        let sigma = eta[1].mapv(|x| self.params[1].link.response(x));

        let mut total_nll = 0.0;
        let y = data.response();
        let w = data.weights();

        for i in 0..data.n_obs() {
            let sig = sigma[i].max(1e-10); // avoid log(0)
            let diff = y[i] - mu[i];

            // Gaussian NLL: 0.5 * log(2 * pi * sigma^2) + 0.5 * ((y - mu) / sigma)^2
            // Equivalently: log(sigma) + 0.5 * log(2pi) + 0.5 * (diff / sigma)^2
            let log_lik = sig.ln() + 0.5 * (2.0 * PI).ln() + 0.5 * (diff / sig).powi(2);

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll += weight * log_lik;
        }

        Ok(total_nll)
    }

    // Override ngradient for analytical score efficiency
    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        let mut grad = Array1::zeros(data.n_obs());
        let y = data.response();
        let w = data.weights();

        let mu = eta[0].mapv(|x| self.params[0].link.response(x));
        let sigma = eta[1].mapv(|x| self.params[1].link.response(x));

        for i in 0..data.n_obs() {
            let sig = sigma[i].max(1e-10);
            let diff = y[i] - mu[i];
            let sig2 = sig * sig;
            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);

            if k == 0 {
                // d(NLL)/d(mu) * d(mu)/d(eta)
                // d(NLL)/d(mu) = - (y - mu) / sigma^2
                // mu link is identity, so deriv is 1. d(mu)/d(eta) = 1 / deriv(mu)
                let d_nll_d_mu = -diff / sig2;
                grad[i] = -weight * d_nll_d_mu / self.params[0].link.deriv(mu[i]);
            } else if k == 1 {
                // d(NLL)/d(sigma) * d(sigma)/d(eta)
                // d(NLL)/d(sigma) = (1 / sigma) - (y - mu)^2 / sigma^3
                let d_nll_d_sigma = (1.0 / sig) - (diff * diff) / (sig2 * sig);
                grad[i] = -weight * d_nll_d_sigma / self.params[1].link.deriv(sigma[i]);
            }
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let mean = weighted_mean(data.response(), data.weights());
        let sd = weighted_sd(data.response(), data.weights()).max(1e-10);

        // Return offsets on the eta scale: [Identity(mean), Log(sd)]
        Ok(vec![
            self.params[0].link.link(mean),
            self.params[1].link.link(sd),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn gaussian_lss_init_offsets() {
        let fam = GaussianLss::new();
        // y = [1.0, 3.0], mean = 2.0, sd = 1.4142...
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None, None).unwrap();
        let offsets = fam.init_offsets(&ds).unwrap();

        assert_relative_eq!(offsets[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(offsets[1], 2.0_f64.sqrt().ln(), epsilon = 1e-4);
    }

    #[test]
    fn gaussian_lss_ngradient_matches_finite_diff() {
        let fam = GaussianLss::new();
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None, None).unwrap();
        let eta = vec![array![0.5, 2.5], array![0.1, -0.2]]; // [mu_eta, sigma_eta]

        let eps = 1e-5;

        for k in 0..2 {
            let analytical_grad = fam.ngradient(&ds, &eta, k).unwrap();

            let mut finite_diff_grad = Array1::zeros(ds.n_obs());
            for i in 0..ds.n_obs() {
                let mut eta_plus = eta.clone();
                let mut eta_minus = eta.clone();

                eta_plus[k][i] += eps;
                eta_minus[k][i] -= eps;

                let l_plus = fam.nll(&ds, &eta_plus).unwrap();
                let l_minus = fam.nll(&ds, &eta_minus).unwrap();

                finite_diff_grad[i] = -(l_plus - l_minus) / (2.0 * eps);
            }

            for i in 0..ds.n_obs() {
                assert_relative_eq!(analytical_grad[i], finite_diff_grad[i], epsilon = 1e-3);
            }
        }
    }
}
