use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, LogitLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_zinb_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", LogLink),
        ParamSpec::new("sigma", LogLink),
        ParamSpec::new("nu", LogitLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZINBLss {
    #[serde(skip, default = "default_zinb_params")]
    params: Vec<ParamSpec>,
}

impl ZINBLss {
    pub fn new() -> Self {
        Self {
            params: default_zinb_params(),
        }
    }
}

impl Default for ZINBLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for ZINBLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let sigma_link = &self.params[1].link;
        let nu_link = &self.params[2].link;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let sigma = sigma_link.response(eta[1][i]).max(1e-10);
            let nu = nu_link.response(eta[2][i]).clamp(1e-10, 1.0 - 1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let var = mu + sigma * mu * mu;
            let p = mu / var;
            let r = mu * mu / (var - mu).max(1e-10);

            let log_pdf = if yi == 0.0 {
                let nb_zero = p.powf(r);
                (nu + (1.0 - nu) * nb_zero).ln()
            } else {
                let ln_gamma_r_y = statrs::function::gamma::ln_gamma(r + yi);
                let ln_gamma_r = statrs::function::gamma::ln_gamma(r);
                let ln_gamma_y_1 = statrs::function::gamma::ln_gamma(yi + 1.0);

                (1.0 - nu).ln() + ln_gamma_r_y - ln_gamma_r - ln_gamma_y_1
                    + r * p.ln()
                    + yi * (1.0 - p).ln()
            };
            nll -= wi * log_pdf;
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(1.0).max(1e-3);
        let sigma_init = 1.0;
        let zeros = y.iter().filter(|&&val| val == 0.0).count();
        let nu_init = (zeros as f64 / y.len() as f64).clamp(0.01, 0.99);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(sigma_init),
            self.params[2].link.link(nu_init),
        ])
    }
}
