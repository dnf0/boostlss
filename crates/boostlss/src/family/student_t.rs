use super::Family;
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;

fn default_student_t_params() -> Vec<ParamSpec> {
    StudentTLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentTLss {
    #[serde(skip, default = "default_student_t_params")]
    params: Vec<ParamSpec>,
}

impl StudentTLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", LogLink), // degrees of freedom
            ],
        }
    }
}

impl Default for StudentTLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for StudentTLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let mu = eta[0].clone(); // Identity link
        let sigma = eta[1].mapv(|x| x.exp()); // Log link
        let nu = eta[2].mapv(|x| x.exp()); // Log link

        let mut total_nll = 0.0;
        let y = data.response();
        let w = data.weights();

        for i in 0..data.n_obs() {
            let sig = sigma[i].max(1e-10);
            let n = nu[i].max(1e-10);
            let diff = y[i] - mu[i];

            // t-distribution NLL:
            // -ln(Gamma((nu+1)/2)) + ln(Gamma(nu/2)) + 0.5*ln(nu*pi) + ln(sigma)
            // + 0.5*(nu+1)*ln(1 + (1/nu)*((y-mu)/sigma)^2)

            let term1 = -ln_gamma((n + 1.0) / 2.0);
            let term2 = ln_gamma(n / 2.0);
            let term3 = 0.5 * (n * PI).ln();
            let term4 = sig.ln();
            let z2 = (diff / sig).powi(2);
            let term5 = 0.5 * (n + 1.0) * (1.0 + z2 / n).ln();

            let log_lik = term1 + term2 + term3 + term4 + term5;
            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll += weight * log_lik;
        }

        Ok(total_nll)
    }

    // We rely on the finite-difference default `ngradient` for StudentT because
    // analytical derivatives involving digamma functions are tedious and error-prone for V1.

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let mean = weighted_mean(data.response(), data.weights());
        let sd = weighted_sd(data.response(), data.weights()).max(1e-10);

        // Return offsets on the eta scale: [Identity(mean), Log(sd), Log(10.0)]
        // Starting with nu=10 is a common stable default for t-distribution boosting.
        Ok(vec![mean, sd.ln(), 10.0_f64.ln()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn student_t_init_offsets() {
        let fam = StudentTLss::new();
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None).unwrap();
        let offsets = fam.init_offsets(&ds).unwrap();

        assert_relative_eq!(offsets[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(offsets[1], 2.0_f64.sqrt().ln(), epsilon = 1e-4);
        assert_relative_eq!(offsets[2], 10.0_f64.ln(), epsilon = 1e-4);
    }
}
