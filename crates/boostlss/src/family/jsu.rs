use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_jsu_params() -> Vec<ParamSpec> {
    JSULss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSULss {
    #[serde(skip, default = "default_jsu_params")]
    params: Vec<ParamSpec>,
}

impl JSULss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", IdentityLink),
                ParamSpec::new("tau", LogLink),
            ],
        }
    }
}

impl Default for JSULss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for JSULss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let mu = eta[0].mapv(|x| self.params[0].link.response(x));
        let sigma = eta[1].mapv(|x| self.params[1].link.response(x));
        let nu = eta[2].mapv(|x| self.params[2].link.response(x));
        let tau = eta[3].mapv(|x| self.params[3].link.response(x));

        let mut total_nll = 0.0;
        let y = data.response();
        let w = data.weights();

        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

        for i in 0..data.n_obs() {
            let sig = sigma[i].max(1e-10);
            let z = (y[i] - mu[i]) / sig;

            // r = -nu + tau * asinh(z)
            let asinh_z = z.asinh();
            let tau_i = tau[i].max(1e-10);
            let r = -nu[i] + tau_i * asinh_z;

            // log_pdf = log(tau) - log(sigma) - 0.5*log(z^2 + 1) - 0.5*log(2*pi) - 0.5*r^2
            let log_pdf =
                tau_i.ln() - sig.ln() - 0.5 * (z * z + 1.0).ln() - half_log_2pi - 0.5 * r * r;

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll -= weight * log_pdf;
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let w = data.weights();

        let mu_init = weighted_mean(y, w);
        let sigma_init = weighted_sd(y, w);

        let nu_init = 0.0;
        let tau_init = 1.0;

        let eta_mu = self.params[0].link.link(mu_init);
        let eta_sigma = self.params[1].link.link(sigma_init.max(1e-10));
        let eta_nu = self.params[2].link.link(nu_init);
        let eta_tau = self.params[3].link.link(tau_init);

        Ok(vec![eta_mu, eta_sigma, eta_nu, eta_tau])
    }

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
        let nu = eta[2].mapv(|x| self.params[2].link.response(x));
        let tau = eta[3].mapv(|x| self.params[3].link.response(x));

        for i in 0..data.n_obs() {
            let sig = sigma[i].max(1e-10);
            let t = tau[i].max(1e-10);

            let z = (y[i] - mu[i]) / sig;
            let z_sq = z * z;
            let z_sq_1 = z_sq + 1.0;
            let sqrt_z_sq_1 = z_sq_1.sqrt();
            let asinh_z = z.asinh();

            // Match the existing `nll` implementation: r = -nu + tau * asinh(z)
            let r = -nu[i] + t * asinh_z;

            let d_nll_d_z = z / z_sq_1 + r * t / sqrt_z_sq_1;

            let d_nll_d_theta = match k {
                0 => {
                    // d_nll / d_mu = d_nll_d_z * d_z_d_mu
                    d_nll_d_z * (-1.0 / sig)
                }
                1 => {
                    // d_nll / d_sigma = 1/sigma + d_nll_d_z * d_z_d_sigma
                    1.0 / sig + d_nll_d_z * (-z / sig)
                }
                2 => {
                    // d_nll / d_nu
                    // r = -nu + tau * asinh(z) => d_r/d_nu = -1
                    // log_pdf has -0.5 * r^2
                    // d_nll / d_r = r
                    // d_nll / d_nu = -r
                    -r
                }
                3 => {
                    // d_nll / d_tau = -1/tau + r * asinh_z
                    -1.0 / t + r * asinh_z
                }
                _ => unreachable!(),
            };

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            let param_val = match k {
                0 => mu[i],
                1 => sigma[i],
                2 => nu[i],
                3 => tau[i],
                _ => unreachable!(),
            };

            // gradient is - d_nll / d_eta = - d_nll / d_theta * (1 / link_deriv)
            grad[i] = -weight * d_nll_d_theta / self.params[k].link.deriv(param_val);
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsu_params() {
        let fam = JSULss::new();
        assert_eq!(fam.params().len(), 4);
        assert_eq!(fam.params()[0].name, "mu");
        assert_eq!(fam.params()[1].name, "sigma");
        assert_eq!(fam.params()[2].name, "nu");
        assert_eq!(fam.params()[3].name, "tau");
    }

    #[test]
    fn test_jsu_nll() {
        use ndarray::{array, Array2};
        let fam = JSULss::new();
        let y = array![0.0, 1.0, 2.0];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
        let eta = vec![
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0],
        ];
        let nll = fam.nll(&ds, &eta).unwrap();
        approx::assert_relative_eq!(nll, 5.3385595, epsilon = 1e-5);
    }

    #[test]
    fn test_jsu_init() {
        use ndarray::{array, Array2};
        let fam = JSULss::new();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, Some(w)).unwrap();

        let offsets = fam.init_offsets(&ds).unwrap();
        assert_eq!(offsets.len(), 4);

        approx::assert_relative_eq!(offsets[0], 3.0, epsilon = 1e-5);
        approx::assert_relative_eq!(offsets[1], 1.5811388f64.ln(), epsilon = 1e-5);
        approx::assert_relative_eq!(offsets[2], 0.0, epsilon = 1e-5);
        approx::assert_relative_eq!(offsets[3], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn jsu_ngradient_matches_finite_diff() {
        use approx::assert_relative_eq;
        let fam = JSULss::new();
        // Use some arbitrary values
        let ds = Dataset::new(
            ndarray::Array2::<f64>::zeros((2, 1)),
            ndarray::array![1.0, 3.0],
            None,
        )
        .unwrap();
        // eta for mu, sigma, nu, tau
        let eta = vec![
            ndarray::array![0.5, 2.5],
            ndarray::array![0.1, -0.2],
            ndarray::array![0.2, 0.1],
            ndarray::array![0.0, 0.1], // log(tau) = 0.0 => tau=1
        ];

        let eps = 1e-5;

        for k in 0..4 {
            let analytical_grad = fam.ngradient(&ds, &eta, k).unwrap();
            let mut finite_diff_grad = ndarray::Array1::zeros(ds.n_obs());

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
