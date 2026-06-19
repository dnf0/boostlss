use crate::engine::{Algorithm, Config, Mstop};
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::BaseLearner;

pub struct BoostLss<F: Family> {
    family: F,
    config: Config,
    learners: Vec<(usize, BaseLearner)>, // (param_index, learner)
}

impl<F: Family> BoostLss<F> {
    pub fn new(family: F) -> Self {
        Self {
            family,
            config: Config::default(),
            learners: Vec::new(),
        }
    }

    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.config.algorithm = algo;
        self
    }

    pub fn mstop(mut self, mstop: Mstop) -> Self {
        self.config.mstop = mstop;
        self
    }

    pub fn step_length(mut self, step: f64) -> Self {
        self.config.step_length = step;
        self
    }

    pub fn family(&self) -> &F {
        &self.family
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn learners(&self) -> &[(usize, BaseLearner)] {
        &self.learners
    }

    /// Registers a base learner for a specific parameter.
    ///
    /// It is supported and intended to register multiple base learners for the same
    /// parameter (e.g. adding both a Linear and a PSpline learner to `mu`).
    pub fn on(mut self, param_name: &str, learner: BaseLearner) -> Result<Self, BoostlssError> {
        let params = self.family.params();
        let k = params
            .iter()
            .position(|p| p.name == param_name)
            .ok_or_else(|| {
                BoostlssError::InvalidConfig(format!("Unknown parameter {}", param_name))
            })?;
        self.learners.push((k, learner));
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::GaussianLss;
    use crate::learner::{BaseLearner, Linear};

    #[test]
    fn test_boostlss_new() {
        let model = BoostLss::new(GaussianLss::new());
        assert_eq!(model.learners().len(), 0);
    }

    #[test]
    fn test_boostlss_on_valid_param() {
        let learner = BaseLearner::Linear(Linear::new("x"));
        let model = BoostLss::new(GaussianLss::new()).on("mu", learner).unwrap();

        assert_eq!(model.learners().len(), 1);
        assert_eq!(model.learners()[0].0, 0);
    }

    #[test]
    fn test_boostlss_on_invalid_param() {
        let learner = BaseLearner::Linear(Linear::new("x"));
        let result = BoostLss::new(GaussianLss::new()).on("invalid_param", learner);

        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }
}
