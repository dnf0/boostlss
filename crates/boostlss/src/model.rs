use crate::data::Dataset;
use crate::engine::{Algorithm, Config, Mstop};
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::{BaseLearner, LearnerUpdate};
use ndarray::Array1;

pub struct ParamBuilder {
    pub(crate) learners: Vec<BaseLearner>,
}

impl ParamBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            learners: Vec::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add<L: Into<BaseLearner>>(mut self, learner: L) -> Self {
        self.learners.push(learner.into());
        self
    }
}

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

    /// Registers base learners for a specific parameter using a builder closure.
    ///
    /// Example:
    /// ```
    /// # use boostlss::model::BoostLss;
    /// # use boostlss::family::GaussianLss;
    /// # use boostlss::learner::{Linear, PSpline};
    /// let model = BoostLss::new(GaussianLss::new())
    ///     .on("mu", |p| p.add(Linear::new("x1")).add(PSpline::new("x2")));
    /// ```
    pub fn on(
        mut self,
        param_name: &str,
        build_fn: impl FnOnce(ParamBuilder) -> ParamBuilder,
    ) -> Result<Self, BoostlssError> {
        let params = self.family.params();
        let k = params
            .iter()
            .position(|p| p.name == param_name)
            .ok_or_else(|| {
                BoostlssError::InvalidConfig(format!("Unknown parameter {}", param_name))
            })?;

        let builder = build_fn(ParamBuilder::new());
        self.learners
            .extend(builder.learners.into_iter().map(|l| (k, l)));
        Ok(self)
    }

    pub fn into_parts(self) -> (F, Config, Vec<(usize, BaseLearner)>) {
        (self.family, self.config, self.learners)
    }

    pub fn fit(self, data: &Dataset) -> Result<Fitted<F>, BoostlssError> {
        match self.config.algorithm {
            Algorithm::Cyclic => crate::engine::cyclical::fit_cyclical(self, data),
            Algorithm::NonCyclic => Err(BoostlssError::InvalidConfig(
                "NonCyclic not yet implemented".into(),
            )),
        }
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
    pub update: LearnerUpdate,
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
            match &update.update {
                LearnerUpdate::Linear(coef) => {
                    let design = learner.build_design(&x_col)?;
                    let u_hat = design.dot(coef);
                    pred = pred + u_hat;
                }
                LearnerUpdate::Stump {
                    split_val,
                    left_val,
                    right_val,
                } => {
                    let u_hat = x_col.mapv(|val| {
                        if val <= *split_val {
                            *left_val
                        } else {
                            *right_val
                        }
                    });
                    pred = pred + u_hat;
                }
                LearnerUpdate::Tree {
                    node: root,
                    param: _,
                } => {
                    if let BaseLearner::Tree(tree_learner) = learner {
                        for i in 0..pred.len() {
                            let mut node_ptr = root;
                            loop {
                                match node_ptr {
                                    crate::learner::TreeNode::Leaf { value, .. } => {
                                        pred[i] += *value;
                                        break;
                                    }
                                    crate::learner::TreeNode::Split {
                                        feature_idx,
                                        threshold,
                                        left,
                                        right,
                                    } => {
                                        let col_idx = tree_learner.feature_indices[*feature_idx];
                                        let val = data.design().column(col_idx)[i];
                                        if val <= *threshold {
                                            node_ptr = left;
                                        } else {
                                            node_ptr = right;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
    use crate::learner::Linear;

    #[test]
    fn test_boostlss_new() {
        let model = BoostLss::new(GaussianLss::new());
        assert_eq!(model.learners().len(), 0);
    }

    #[test]
    fn test_boostlss_on_valid_param() {
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap();

        assert_eq!(model.learners().len(), 1);
        assert_eq!(model.learners()[0].0, 0);
    }

    #[test]
    fn test_boostlss_on_invalid_param() {
        let result =
            BoostLss::new(GaussianLss::new()).on("invalid_param", |p| p.add(Linear::new("x")));

        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }

    #[test]
    fn test_boostlss_on_multiple_learners() {
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| {
                p.add(Linear::new("x"))
                    .add(crate::learner::PSpline::new("y"))
            })
            .unwrap();

        assert_eq!(model.learners().len(), 2);
        assert_eq!(model.learners()[0].0, 0);
        assert_eq!(model.learners()[1].0, 0);
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
