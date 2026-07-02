use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_merton_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("lam", LogLink),
        ParamSpec::new("mu_j", IdentityLink),
        ParamSpec::new("sigma_j", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MertonJumpDiffusionLss {
    pub max_jumps: usize,
    #[serde(skip, default = "default_merton_params")]
    params: Vec<ParamSpec>,
}

impl MertonJumpDiffusionLss {
    pub fn new(max_jumps: usize) -> Self {
        Self {
            max_jumps,
            params: default_merton_params(),
        }
    }
}

impl Default for MertonJumpDiffusionLss {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Family for MertonJumpDiffusionLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let lam_link = &self.params[2].link;
        let mu_j_link = &self.params[3].link;
        let sigma_j_link = &self.params[4].link;
        let w = data.weights();

        let mut total_nll = 0.0;
        let pi = std::f64::consts::PI;

        // Pre-compute factorial log for logsumexp trick
        let mut ln_fact = vec![0.0; self.max_jumps + 1];
        for j in 1..=self.max_jumps {
            ln_fact[j] = ln_fact[j - 1] + (j as f64).ln();
        }

        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let lam = lam_link.response(eta[2][i]).max(1e-10);
            let mu_j = mu_j_link.response(eta[3][i]);
            let sigma_j = sigma_j_link.response(eta[4][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let var_diff = sigma * sigma;
            let var_jump = sigma_j * sigma_j;
            let drift = mu - 0.5 * var_diff;

            // Use LogSumExp trick for stability
            let mut log_terms = Vec::with_capacity(self.max_jumps + 1);
            for (j, ln_fact_j) in ln_fact.iter().enumerate().take(self.max_jumps + 1) {
                let j_f64 = j as f64;
                let mu_total = drift + j_f64 * mu_j;
                let var_total = var_diff + j_f64 * var_jump;
                let std_total = var_total.sqrt();

                // ln_prob_jump = -lam + j*ln(lam) - ln(j!)
                let ln_prob_jump = -lam + j_f64 * lam.ln() - ln_fact_j;

                let diff = yi - mu_total;
                let ln_norm =
                    -0.5 * (2.0 * pi).ln() - std_total.ln() - 0.5 * (diff * diff) / var_total;

                log_terms.push(ln_prob_jump + ln_norm);
            }

            // logsumexp
            let max_log = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            for lt in log_terms {
                sum_exp += (lt - max_log).exp();
            }
            let ln_likelihood = max_log + sum_exp.ln();

            total_nll -= wi * ln_likelihood;
        }
        Ok(total_nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let sigma_init = y.var(1.0).sqrt().max(1e-3);
        let lam_init = 1.0;
        let mu_j_init = 0.0;
        let sigma_j_init = sigma_init; // Initialize jump volatility same as diffusion

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(lam_init),
            self.params[3].link.link(mu_j_init),
            self.params[4].link.link(sigma_j_init),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_merton_init() {
        let fam = MertonJumpDiffusionLss::new(10);
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let ds = Dataset::new(Array2::<f64>::zeros((5, 1)), y, Some(w), None).unwrap();

        let offsets = fam.init_offsets(&ds).unwrap();
        assert_eq!(offsets.len(), 5);
        assert_eq!(offsets[0], 3.0); // mean
        assert!(offsets[1] > 0.0); // log(var.sqrt())
    }
}
