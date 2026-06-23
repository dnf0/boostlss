use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, LogitLink, ParamSpec};
use crate::util::{minimize_1d, weighted_mean};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::digamma;

const EPSILON: f64 = 1e-10;

fn default_beta_params() -> Vec<ParamSpec> {
    BetaLss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaLss {
    #[serde(skip, default = "default_beta_params")]
    params: Vec<ParamSpec>,
}

impl BetaLss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", LogitLink),
                ParamSpec::new("phi", LogLink),
            ],
        }
    }

    fn check_response(&self, y: &Array1<f64>) -> Result<(), BoostlssError> {
        if y.iter().any(|&val| val <= 0.0 || val >= 1.0) {
            return Err(BoostlssError::DataError(
                "Beta response must be strictly between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for BetaLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for BetaLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        self.check_response(data.response())?;
        let y = data.response();
        let mu_link = &self.params[0].link;
        let phi_link = &self.params[1].link;

        let mut nll = 0.0;
        let w = data.weights();

        for i in 0..y.len() {
            let yi = y[i];
            let mui = eta[0][i];
            let phii = eta[1][i];
            let wi = w.map_or(1.0, |weights| weights[i]);

            let mu = mu_link.response(mui).clamp(EPSILON, 1.0 - EPSILON);
            let phi = phi_link.response(phii).max(EPSILON);

            let alpha = mu * phi;
            let beta = (1.0 - mu) * phi;

            // log(Gamma(alpha + beta) / (Gamma(alpha) * Gamma(beta))) + (alpha - 1)log(y) + (beta - 1)log(1-y)
            let log_pdf = statrs::function::gamma::ln_gamma(alpha + beta)
                - statrs::function::gamma::ln_gamma(alpha)
                - statrs::function::gamma::ln_gamma(beta)
                + (alpha - 1.0) * yi.ln()
                + (beta - 1.0) * (1.0 - yi).ln();

            nll -= wi * log_pdf;
        }
        Ok(nll)
    }

    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let phi_link = &self.params[1].link;
        let mut grad = Array1::zeros(y.len());

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).clamp(EPSILON, 1.0 - EPSILON);
            let phi = phi_link.response(eta[1][i]).max(EPSILON);

            let alpha = mu * phi;
            let beta = (1.0 - mu) * phi;
            let wi = data.weights().map_or(1.0, |w| w[i]);

            if k == 0 {
                // ngradient w.r.t mu predictor
                let d_l_d_mu = phi * (yi.ln() - (1.0 - yi).ln() - digamma(alpha) + digamma(beta));
                let d_mu_d_eta = 1.0 / mu_link.deriv(mu);
                grad[i] = wi * d_l_d_mu * d_mu_d_eta;
            } else {
                // ngradient w.r.t phi predictor
                let d_l_d_phi = mu * (yi.ln() - digamma(alpha))
                    + (1.0 - mu) * ((1.0 - yi).ln() - digamma(beta))
                    + digamma(alpha + beta);
                let d_phi_d_eta = 1.0 / phi_link.deriv(phi);
                grad[i] = wi * d_l_d_phi * d_phi_d_eta;
            }
        }
        Ok(grad)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let w = data.weights();
        let mu_hat = weighted_mean(y, w).clamp(EPSILON, 1.0 - EPSILON);

        // Refine with 1D optimization for phi
        let y_arr = y.clone();
        let w_arr = w.cloned();
        let dummy_design = Array2::zeros((y.len(), 0));
        let dummy_dataset = Dataset::new(dummy_design, y_arr.clone(), w_arr).unwrap();

        let opt_phi = minimize_1d(
            |log_phi| {
                let eta = vec![
                    Array1::from_elem(y_arr.len(), self.params[0].link.link(mu_hat)),
                    Array1::from_elem(y_arr.len(), log_phi),
                ];
                self.nll(&dummy_dataset, &eta).unwrap_or(f64::MAX)
            },
            -5.0,
            10.0,
        );

        Ok(vec![self.params[0].link.link(mu_hat), opt_phi])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_beta_gradients() {
        let fam = BetaLss::new();
        let y = array![0.2, 0.5, 0.8];
        let ds = Dataset::new(Array2::<f64>::zeros((3, 1)), y, None).unwrap();
        let eta = vec![array![-1.0, 0.0, 1.0], array![0.0, 1.0, 2.0]]; // logit(mu), log(phi)

        // Compare analytic gradient with finite difference
        let grad_mu_analytic = fam.ngradient(&ds, &eta, 0).unwrap();
        let grad_phi_analytic = fam.ngradient(&ds, &eta, 1).unwrap();

        // We use the default trait method for finite diffs by implementing a dummy struct wrapper if needed,
        // but here we just manually do the finite diff for test validation
        let eps = 1e-5;
        let mut eta_plus = eta.clone();
        let mut eta_minus = eta.clone();

        // check mu
        for i in 0..3 {
            eta_plus[0][i] += eps;
            eta_minus[0][i] -= eps;
            let l_p = fam.nll(&ds, &eta_plus).unwrap();
            let l_m = fam.nll(&ds, &eta_minus).unwrap();
            let fin_diff = -(l_p - l_m) / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_mu_analytic[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
            eta_plus[0][i] -= eps;
            eta_minus[0][i] += eps;
        }

        // check phi
        for i in 0..3 {
            eta_plus[1][i] += eps;
            eta_minus[1][i] -= eps;
            let l_p = fam.nll(&ds, &eta_plus).unwrap();
            let l_m = fam.nll(&ds, &eta_minus).unwrap();
            let fin_diff = -(l_p - l_m) / (2.0 * eps);
            assert!(approx::relative_eq!(
                grad_phi_analytic[i],
                fin_diff,
                epsilon = 1e-4,
                max_relative = 1e-3
            ));
            eta_plus[1][i] -= eps;
            eta_minus[1][i] += eps;
        }
    }
}
