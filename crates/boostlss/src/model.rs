use crate::data::Dataset;
use crate::engine::{Algorithm, Config, Mstop};
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::BaseLearner;
use ndarray::Array1;

#[derive(Clone)]
pub struct BoostLss<F: Family + Clone> {
    family: F,
    config: Config,
    learners: Vec<(usize, BaseLearner)>, // (param_index, learner)
}

impl<F: Family + Clone> BoostLss<F> {
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

    pub fn into_parts(self) -> (F, Config, Vec<(usize, BaseLearner)>) {
        (self.family, self.config, self.learners)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Link,
    Response,
}

#[derive(Debug, Clone)]
pub struct UpdateStep {
    pub param_idx: usize,
    pub learner_idx: usize,
    pub coef: Array1<f64>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Fitted<F: Family> {
    family: F,
    offsets: Vec<f64>,
    /// Accumulated fits for each parameter's learners over iterations
    /// Will store the selected sequence of updates.
    pub updates: Vec<UpdateStep>,
    /// The base learners used during fitting, needed for prediction
    learners: Vec<(usize, BaseLearner)>,
}

impl<F: Family> Fitted<F> {
    pub fn new(family: F, offsets: Vec<f64>, learners: Vec<(usize, BaseLearner)>) -> Self {
        Self {
            family,
            offsets,
            updates: Vec::new(),
            learners,
        }
    }

    pub fn predict(
        &mut self,
        data: &Dataset,
        param: &str,
        scale: Scale,
    ) -> Result<Array1<f64>, BoostlssError> {
        let params = self.family.params();
        let k = params
            .iter()
            .position(|p| p.name == param)
            .ok_or_else(|| BoostlssError::InvalidConfig(format!("Unknown parameter {}", param)))?;

        let mut pred = Array1::from_elem(data.n_obs(), self.offsets[k]);
        let x_col = data.design().column(0).to_owned();

        for update in &self.updates {
            if update.param_idx != k {
                continue;
            }

            let learner = &mut self.learners[update.learner_idx].1;
            let design = learner.build_design(&x_col)?;
            let u_hat = design.dot(&update.coef);
            pred = pred + u_hat;
        }

        match scale {
            Scale::Link => Ok(pred),
            Scale::Response => Ok(pred.mapv(|x| params[k].link.response(x))),
        }
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

    #[test]
    fn test_fitted_new_and_predict() {
        use ndarray::{Array1, Array2};
        let family = GaussianLss::new();
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], vec![]);
        let data = Dataset::new(Array2::zeros((5, 2)), Array1::zeros(5), None).unwrap();

        let pred = fitted.predict(&data, "mu", Scale::Link).unwrap();
        assert_eq!(pred.len(), 5);
        assert_eq!(pred, Array1::zeros(5));
    }
}
