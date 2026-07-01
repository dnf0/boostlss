use super::Family;
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::{LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

const EPSILON: f64 = 1e-10;

fn default_gamma_params() -> Vec<ParamSpec> {
    GammaLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaLss {
    #[serde(skip, default = "default_gamma_params")]
    params: Vec<ParamSpec>,
}

impl GammaLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogLink),
                ParamSpec::new("sigma", LogLink), // coefficient of variation
            ],
        }
    }
}

impl Default for GammaLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for GammaLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let mu = eta[0].mapv(|x| x.exp());
        let sigma = eta[1].mapv(|x| x.exp());

        let mut total_nll = 0.0;
        let y = data.response();
        let w = data.weights();

        for i in 0..data.n_obs() {
            let m = mu[i].max(EPSILON);
            let s = sigma[i].max(EPSILON);
            let yi = y[i].max(EPSILON); // Response must be strictly positive

            // Gamma distribution parameterized by mean (mu) and coeff of var (sigma)
            // shape (alpha) = 1 / sigma^2
            // rate (beta) = 1 / (mu * sigma^2)

            let alpha = 1.0 / (s * s);
            let beta = alpha / m;

            // NLL = - (alpha * ln(beta) - ln_gamma(alpha) + (alpha - 1)*ln(yi) - beta * yi)
            let log_lik = alpha * beta.ln() - ln_gamma(alpha) + (alpha - 1.0) * yi.ln() - beta * yi;

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll += weight * -log_lik; // Negate to get NLL
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let mean = weighted_mean(data.response(), data.weights());
        let sd = weighted_sd(data.response(), data.weights()).max(EPSILON);

        // For gamma, sigma is the coefficient of variation (sd / mean)
        let cv = sd / mean.max(EPSILON);

        Ok(vec![mean.max(EPSILON).ln(), cv.max(EPSILON).ln()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn gamma_init_offsets() {
        let fam = GammaLss::new();
        // y = [1.0, 3.0], mean = 2.0, sd = 1.4142...
        // cv = 1.4142... / 2.0 = 0.7071...
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None, None).unwrap();
        let offsets = fam.init_offsets(&ds).unwrap();

        assert_relative_eq!(offsets[0], 2.0_f64.ln(), epsilon = 1e-4);
        assert_relative_eq!(offsets[1], (2.0_f64.sqrt() / 2.0).ln(), epsilon = 1e-4);
    }

    #[test]
    fn gamma_nll_is_accurate() {
        let fam = GammaLss::new();
        let ds = Dataset::new(Array2::<f64>::zeros((1, 1)), array![2.0], None, None).unwrap();

        // Set mu = 2.0, sigma = 0.5
        let eta = vec![array![2.0_f64.ln()], array![0.5_f64.ln()]];
        let nll = fam.nll(&ds, &eta).unwrap();

        // Expected NLL for y=2, mu=2, sigma=0.5:
        // alpha = 1 / 0.5^2 = 4.0
        // beta = alpha / mu = 4.0 / 2.0 = 2.0
        // log_lik = 4.0*ln(2) - ln_gamma(4) + 3.0*ln(2) - 4.0 = ~ -0.93973
        // nll = 0.93973
        assert_relative_eq!(nll, 0.93973, epsilon = 1e-4);
    }
}
