use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_logistic_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("s", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticLss {
    #[serde(skip, default = "default_logistic_params")]
    params: Vec<ParamSpec>,
}

impl LogisticLss {
    pub fn new() -> Self {
        Self {
            params: default_logistic_params(),
        }
    }
}

impl Default for LogisticLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for LogisticLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let s_link = &self.params[1].link;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let s = s_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            let z = (yi - mu) / s;
            nll += wi * (z + s.ln() + 2.0 * (1.0 + (-z).exp()).ln());
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();
        let mu_init = y.mean().unwrap_or(0.0);
        let var = y.var(1.0);
        let s_init = (var * 3.0 / std::f64::consts::PI.powi(2)).sqrt().max(1e-3);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(s_init),
        ])
    }
}
