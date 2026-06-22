use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogitLink, ParamSpec};
use crate::util::weighted_mean;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_binomial_params() -> Vec<ParamSpec> {
    BinomialLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinomialLss {
    #[serde(skip, default = "default_binomial_params")]
    params: Vec<ParamSpec>,
}

impl BinomialLss {
    pub fn new() -> Self {
        Self {
            params: vec![ParamSpec::new("mu", LogitLink)],
        }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)) {
            Ok(())
        } else {
            Err(BoostlssError::DataError(
                "BinomialLss requires 0.0 <= y <= 1.0".into(),
            ))
        }
    }
}

impl Default for BinomialLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for BinomialLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();

        let mut total_nll = 0.0;
        let w = data.weights();
        let eta_mu = &eta[0];

        for i in 0..data.n_obs() {
            let eta_i = eta_mu[i];
            let y_i = y[i];

            // Stable NLL computation: log(1 + exp(eta)) - y * eta
            // For large positive eta, log(1 + exp(eta)) ≈ eta
            let log_1_plus_exp = if eta_i > 0.0 {
                eta_i + (-eta_i).exp().ln_1p()
            } else {
                eta_i.exp().ln_1p()
            };

            let nll = log_1_plus_exp - y_i * eta_i;

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll += weight * nll;
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
                "BinomialLss has 1 parameter".into(),
            ));
        }

        let y = data.response();
        let w = data.weights();
        let eta_mu = &eta[0];

        // Note: Analytical gradient simplification relies explicitly on LogitLink
        let mu = eta_mu.mapv(|x| self.params[0].link.response(x));
        let mut grad = y - &mu;

        if let Some(w_arr) = w {
            grad *= w_arr;
        }

        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;

        let mean = weighted_mean(data.response(), data.weights());
        let clamped = mean.clamp(1e-5, 1.0 - 1e-5);
        Ok(vec![self.params[0].link.link(clamped)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn check_response_bounds() {
        let fam = BinomialLss::new();
        assert!(fam.check_response(&array![0.0, 0.5, 1.0]).is_ok());
        assert!(fam.check_response(&array![-0.1, 0.5, 1.0]).is_err());
        assert!(fam.check_response(&array![0.0, 0.5, 1.1]).is_err());
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let fam = BinomialLss::new();
        let y = array![0.0, 1.0, 0.3, 0.7, 1.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, None).unwrap();
        let etas = vec![array![-2.0, 0.0, 1.5, 5.0, -10.0]];

        let analytical_grad = fam.ngradient(&ds, &etas, 0).unwrap();
        let eps = 1e-5;

        let mut finite_diff_grad = Array1::zeros(ds.n_obs());
        for i in 0..ds.n_obs() {
            let mut eta_plus = etas.clone();
            let mut eta_minus = etas.clone();

            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;

            let l_plus = fam.nll(&ds, &eta_plus).unwrap();
            let l_minus = fam.nll(&ds, &eta_minus).unwrap();

            finite_diff_grad[i] = -(l_plus - l_minus) / (2.0 * eps);
            assert_relative_eq!(analytical_grad[i], finite_diff_grad[i], epsilon = 1e-3);
        }
    }
}
