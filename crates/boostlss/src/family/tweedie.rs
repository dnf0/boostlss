use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_tweedie_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", LogLink),
        ParamSpec::new("phi", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweedieLss {
    pub p: f64,
    #[serde(skip, default = "default_tweedie_params")]
    params: Vec<ParamSpec>,
}

impl TweedieLss {
    pub fn new(p: f64) -> Self {
        assert!(p > 1.0 && p < 2.0, "Tweedie p must be in (1, 2)");
        Self {
            p,
            params: default_tweedie_params(),
        }
    }
}

impl Default for TweedieLss {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl Family for TweedieLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let phi_link = &self.params[1].link;
        let p = self.p;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i].max(0.0);
            let mu = mu_link.response(eta[0][i]).max(1e-10);
            let phi = phi_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            // Using Tweedie Deviance approximation for NLL
            // d(y, mu) = 2 * ( y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )
            let p1 = 1.0 - p;
            let p2 = 2.0 - p;

            let term1 = if yi > 0.0 {
                yi.powf(p2) / (p1 * p2)
            } else {
                0.0
            };
            let term2 = yi * mu.powf(p1) / p1;
            let term3 = mu.powf(p2) / p2;

            let deviance = 2.0 * (term1 - term2 + term3);

            // Approximate NLL: 0.5 * deviance / phi + 0.5 * ln(phi)
            nll += wi * (0.5 * deviance / phi + 0.5 * phi.ln());
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(1.0).max(1e-3);
        let phi_init = 1.0;
        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(phi_init),
        ])
    }
}
