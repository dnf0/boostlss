use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use crate::util::weighted_mean;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

fn default_poisson_params() -> Vec<ParamSpec> {
    PoissonLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonLss {
    #[serde(skip, default = "default_poisson_params")]
    params: Vec<ParamSpec>,
}

impl PoissonLss {
    pub fn new() -> Self {
        Self {
            params: vec![ParamSpec::new("mu", LogLink)],
        }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val < 0.0 || val.fract() != 0.0) {
            return Err(BoostlssError::DataError(
                "Poisson response must be non-negative integers".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for PoissonLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for PoissonLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let w = data.weights();
        let mu_link = &self.params[0].link;

        let mut total_nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let mu = mu_link.response(eta[0][i]).max(1e-10);

            let nll = mu - yi * mu.ln() + ln_gamma(yi + 1.0);
            total_nll += wi * nll;
        }

        Ok(total_nll)
    }

    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        if k != 0 {
            return Err(BoostlssError::InvalidConfig(
                "Poisson only has 1 param".into(),
            ));
        }
        let y = data.response();
        let w = data.weights();
        let mu_link = &self.params[0].link;
        let mut grad = Array1::zeros(y.len());

        for i in 0..y.len() {
            let yi = y[i];
            let wi = w.map_or(1.0, |weights| weights[i]);
            let mu = mu_link.response(eta[0][i]).max(1e-10);

            // d_l_d_mu = 1 - y/mu
            // d_mu_d_eta = mu
            // d_l_d_eta = mu - y
            // Negative gradient is y - mu
            grad[i] = wi * (yi - mu);
        }

        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;
        let mean = weighted_mean(data.response(), data.weights());
        Ok(vec![self.params[0].link.link(mean.max(1e-5))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_poisson_gradients() {
        let fam = PoissonLss::new();
        let y = array![0.0, 1.0, 2.0, 5.0];
        let ds = Dataset::new(Array2::<f64>::zeros((4, 1)), y, None, None).unwrap();
        let eta = vec![array![-1.0, 0.0, 1.0, 2.0]];

        let grad = fam.ngradient(&ds, &eta, 0).unwrap();
        let eps = 1e-5;

        for i in 0..4 {
            let mut eta_plus = eta.clone();
            let mut eta_minus = eta.clone();
            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;

            let fin_diff = -(fam.nll(&ds, &eta_plus).unwrap() - fam.nll(&ds, &eta_minus).unwrap())
                / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
        }
    }
}
