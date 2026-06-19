use super::Family;
use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::{LogLink, ParamSpec};
use crate::util::{weighted_mean, weighted_sd};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;

#[derive(Debug)]
pub struct GammaLss {
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
            let m = mu[i].max(1e-10);
            let s = sigma[i].max(1e-10);
            let yi = y[i].max(1e-10); // Response must be strictly positive

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
        let sd = weighted_sd(data.response(), data.weights()).max(1e-10);

        // For gamma, sigma is the coefficient of variation (sd / mean)
        let cv = sd / mean.max(1e-10);

        Ok(vec![mean.max(1e-10).ln(), cv.max(1e-10).ln()])
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
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None).unwrap();
        let offsets = fam.init_offsets(&ds).unwrap();

        assert_relative_eq!(offsets[0], 2.0_f64.ln(), epsilon = 1e-4);
        assert_relative_eq!(offsets[1], (2.0_f64.sqrt() / 2.0).ln(), epsilon = 1e-4);
    }
}
