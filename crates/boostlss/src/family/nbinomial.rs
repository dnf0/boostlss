use super::Family;
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::{LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;

const EPSILON: f64 = 1e-10;

#[derive(Debug)]
pub struct NBinomialLss {
    params: Vec<ParamSpec>,
}

impl NBinomialLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogLink),
                ParamSpec::new("sigma", LogLink), // dispersion parameter
            ],
        }
    }
}

impl Default for NBinomialLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for NBinomialLss {
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
            let yi = y[i].max(0.0); // Count data, must be >= 0

            // Negative Binomial NLL parameterization (mu, sigma)
            // Var(Y) = mu + sigma * mu^2
            // r = 1 / sigma
            // p = 1 / (1 + sigma * mu)

            let r = 1.0 / s;
            let log_lik = ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
                + r * (r / (r + m)).ln()
                + yi * (m / (r + m)).ln();

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll += weight * -log_lik;
        }

        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let mean = weighted_mean(data.response(), data.weights());
        let sd = weighted_sd(data.response(), data.weights()).max(EPSILON);

        let var = sd * sd;
        // Method of moments for sigma: sigma = (Var - mu) / mu^2
        // If Var <= mu, underdispersion, fallback to small sigma (e.g., 0.1)
        let sigma = if var > mean {
            (var - mean) / (mean * mean).max(EPSILON)
        } else {
            0.1
        };

        Ok(vec![mean.max(EPSILON).ln(), sigma.max(EPSILON).ln()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn nbinomial_init_offsets() {
        let fam = NBinomialLss::new();
        // y = [1.0, 5.0], mean = 3.0, sd = 2.8284, var = 8.0
        // sigma = (8.0 - 3.0) / 9.0 = 5.0 / 9.0 = 0.5555...
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 5.0], None).unwrap();
        let offsets = fam.init_offsets(&ds).unwrap();

        assert_relative_eq!(offsets[0], 3.0_f64.ln(), epsilon = 1e-4);
        assert_relative_eq!(offsets[1], (5.0_f64 / 9.0).ln(), epsilon = 1e-4);
    }

    #[test]
    fn nbinomial_nll_is_accurate() {
        let fam = NBinomialLss::new();
        let ds = Dataset::new(Array2::<f64>::zeros((1, 1)), array![5.0], None).unwrap();

        // Set mu = 3.0, sigma = 0.5
        let eta = vec![array![3.0_f64.ln()], array![0.5_f64.ln()]];
        let nll = fam.nll(&ds, &eta).unwrap();

        // Expected NLL for y=5, mu=3, sigma=0.5:
        // r = 1/0.5 = 2.0
        // log_lik = ln_gamma(7) - ln_gamma(2) - ln_gamma(6) + 2*ln(2/5) + 5*ln(3/5)
        // log_lik = ~ -2.59495
        // NLL = 2.59495
        assert_relative_eq!(nll, 2.59495, epsilon = 1e-4);
    }
}
