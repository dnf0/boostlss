use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::param::{IdentityLink, LogLink, ParamSpec};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

fn default_jsu_params() -> Vec<ParamSpec> {
    JSULss::new().params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSULss {
    #[serde(skip, default = "default_jsu_params")]
    pub params: Vec<ParamSpec>,
}

impl JSULss {
    pub fn new() -> Self {
        Self {
            params: vec![
                ParamSpec::new("mu", IdentityLink),
                ParamSpec::new("sigma", LogLink),
                ParamSpec::new("nu", IdentityLink),
                ParamSpec::new("tau", LogLink),
            ],
        }
    }
}

impl Default for JSULss {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for JSULss {
    fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    fn nll(&self, _data: &Dataset, _eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
        unimplemented!()
    }

    fn init_offsets(&self, _data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsu_params() {
        let fam = JSULss::new();
        assert_eq!(fam.params().len(), 4);
        assert_eq!(fam.params()[0].name, "mu");
        assert_eq!(fam.params()[1].name, "sigma");
        assert_eq!(fam.params()[2].name, "nu");
        assert_eq!(fam.params()[3].name, "tau");
    }
}
