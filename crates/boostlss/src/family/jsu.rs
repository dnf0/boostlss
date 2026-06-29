use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{minimize_1d, weighted_mean, weighted_sd};
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
        unimplemented!()
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
}
