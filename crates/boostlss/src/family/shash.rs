use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_shash_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("nu", IdentityLink),
        ParamSpec::new("tau", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHASHLss {
    #[serde(skip, default = "default_shash_params")]
    params: Vec<ParamSpec>,
}

impl SHASHLss {
    pub fn new() -> Self {
        Self {
            params: default_shash_params(),
        }
    }
}

impl Default for SHASHLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for SHASHLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;
        let tau_link = &self.params[3].link;
        let w = data.weights();

        let mut total_nll = 0.0;
        let log_2pi_half = 0.5 * std::f64::consts::TAU.ln();

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]);
            let tau = tau_link.response(eta[3][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let z = (yi - mu) / sigma;
            let asinh_z = z.asinh();

            let term1 = (tau * asinh_z).exp();
            let term2 = (-nu * asinh_z).exp();

            let r = 0.5 * (term1 - term2);
            let c = 0.5 * (tau * term1 + nu * term2);

            let c_safe = c.max(1e-15);

            let log_likelihood =
                c_safe.ln() - log_2pi_half - sigma.ln() - 0.5 * (1.0 + z * z).ln() - 0.5 * (r * r);

            total_nll -= wi * log_likelihood;
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let sigma_init = y.var(1.0).sqrt().max(1e-3);
        let nu_init = 0.0;
        let tau_init = 1.0;

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(nu_init),
            self.params[3].link.link(tau_init),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_shash_init() {
        let fam = SHASHLss::new();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, Some(w), None).unwrap();

        let offsets = fam.init_offsets(&ds).unwrap();
        assert_eq!(offsets.len(), 4);
        assert_eq!(offsets[0], 3.0); // mean
        assert!(offsets[1] > 0.0); // log(var.sqrt())
        assert_eq!(offsets[2], 0.0);
        assert_eq!(offsets[3], 0.0); // ln(1.0)
    }
}
