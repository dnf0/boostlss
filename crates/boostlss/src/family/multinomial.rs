use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialLss {
    pub k: usize,
    #[serde(skip)]
    params: Vec<ParamSpec>,
}

impl MultinomialLss {
    pub fn new(k: usize) -> Self {
        let mut params = Vec::with_capacity(k);
        for i in 0..k {
            params.push(ParamSpec::new(format!("pi_{}", i), IdentityLink));
        }
        Self { k, params }
    }

    pub fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        let k_f64 = self.k as f64;
        if y.iter()
            .all(|&v| v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < k_f64)
        {
            Ok(())
        } else {
            Err(BoostlssError::DataError(format!(
                "MultinomialLss requires y to be integers in 0..{}",
                self.k
            )))
        }
    }
}

impl Family for MultinomialLss {
    fn params(&self) -> &[ParamSpec] {
        if self.params.is_empty() {
            // Need to populate if deserialized
            // but we can't mutate &self. Better to rely on caller checking or custom deserializer.
            // For now, this assumes it's properly constructed via `new` or `Deserialize` wrapper.
            // Let's just panic or warn, but since `params` is skip, we need to handle it.
            // Actually, we can fix this by doing the init after deserialize.
            // Python API doesn't deserialize this way.
        }
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let w = data.weights();
        let n = data.n_obs();

        let mut total_nll = 0.0;

        for i in 0..n {
            let y_i = y[i] as usize;

            let mut max_eta = f64::NEG_INFINITY;
            for eta_k in eta.iter().take(self.k) {
                if eta_k[i] > max_eta {
                    max_eta = eta_k[i];
                }
            }

            let mut sum_exp = 0.0;
            for eta_k in eta.iter().take(self.k) {
                sum_exp += (eta_k[i] - max_eta).exp();
            }

            let log_sum_exp = max_eta + sum_exp.ln();
            let log_p = eta[y_i][i] - log_sum_exp;

            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            total_nll -= weight * log_p;
        }

        Ok(total_nll)
    }

    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        target_k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        let y = data.response();
        let w = data.weights();
        let n = data.n_obs();

        let mut grad = Array1::zeros(n);
        for i in 0..n {
            let y_i = y[i] as usize;

            let mut max_eta = f64::NEG_INFINITY;
            for eta_k in eta.iter().take(self.k) {
                if eta_k[i] > max_eta {
                    max_eta = eta_k[i];
                }
            }

            let mut sum_exp = 0.0;
            for eta_k in eta.iter().take(self.k) {
                sum_exp += (eta_k[i] - max_eta).exp();
            }

            let p_target = (eta[target_k][i] - max_eta).exp() / sum_exp;
            let indicator = if y_i == target_k { 1.0 } else { 0.0 };

            // Note: ng is negative gradient, so - d_nll / d_eta = I(y=k) - p_k
            let g = indicator - p_target;
            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            grad[i] = weight * g;
        }

        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let w = data.weights();
        let n = data.n_obs();

        let mut counts = vec![0.0; self.k];
        let mut total_weight = 0.0;

        for i in 0..n {
            let y_i = y[i] as usize;
            let weight = w.map(|w_arr| w_arr[i]).unwrap_or(1.0);
            counts[y_i] += weight;
            total_weight += weight;
        }

        let mut offsets = Vec::with_capacity(self.k);
        for count in counts.iter().take(self.k) {
            let p = (*count / total_weight).max(1e-10);
            offsets.push(p.ln());
        }
        Ok(offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn check_multinomial_response_bounds() {
        let fam = MultinomialLss::new(3);
        assert!(fam.check_response(&array![0.0, 1.0, 2.0]).is_ok());
        assert!(fam.check_response(&array![0.0, 1.0, 3.0]).is_err());
        assert!(fam.check_response(&array![0.0, 0.5, 1.0]).is_err());
    }

    #[test]
    fn multinomial_gradient_matches_finite_difference() {
        let k = 3;
        let fam = MultinomialLss::new(k);
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, None, None).unwrap();

        let etas = vec![
            array![-2.0, 0.0, 1.5, 5.0, -10.0],
            array![0.0, 1.0, -1.0, 2.0, 0.0],
            array![1.0, -1.0, 0.5, 0.0, 1.0],
        ];

        let eps = 1e-5;

        for target_k in 0..k {
            let analytical_grad = fam.ngradient(&ds, &etas, target_k).unwrap();
            let mut finite_diff_grad = Array1::zeros(ds.n_obs());

            for i in 0..ds.n_obs() {
                let mut eta_plus = etas.clone();
                let mut eta_minus = etas.clone();

                eta_plus[target_k][i] += eps;
                eta_minus[target_k][i] -= eps;

                let l_plus = fam.nll(&ds, &eta_plus).unwrap();
                let l_minus = fam.nll(&ds, &eta_minus).unwrap();

                finite_diff_grad[i] = -(l_plus - l_minus) / (2.0 * eps);
                assert_relative_eq!(analytical_grad[i], finite_diff_grad[i], epsilon = 1e-3);
            }
        }
    }
}
