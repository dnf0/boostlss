use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_laplace_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("mu", IdentityLink),
        ParamSpec::new("b", LogLink),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceLss {
    #[serde(skip, default = "default_laplace_params")]
    params: Vec<ParamSpec>,
}

impl LaplaceLss {
    pub fn new() -> Self {
        Self {
            params: default_laplace_params(),
        }
    }
}

impl Default for LaplaceLss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for LaplaceLss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        let y = data.response();
        let mu_link = &self.params[0].link;
        let b_link = &self.params[1].link;
        let w = data.weights();

        let mut nll = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mu = mu_link.response(eta[0][i]);
            let b = b_link.response(eta[1][i]).max(1e-10);
            let wi = w.map_or(1.0, |weights| weights[i]);

            nll += wi * (b.ln() + (yi - mu).abs() / b);
        }
        Ok(nll)
    }

    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        let y = data.response();

        let mut sorted_y = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mu_init = sorted_y[sorted_y.len() / 2]; // median

        let sum_abs_dev: f64 = y.iter().map(|&yi| (yi - mu_init).abs()).sum();
        let b_init = (sum_abs_dev / y.len() as f64).max(1e-3);

        Ok(vec![
            self.params[0].link.link(mu_init),
            self.params[1].link.link(b_init),
        ])
    }
}
