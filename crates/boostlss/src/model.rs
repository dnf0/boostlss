use crate::data::Dataset;
use crate::engine::{Algorithm, Config, Mstop};
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::{BaseLearner, LearnerUpdate};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

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

#[derive(Clone, Serialize, Deserialize)]
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
    ///     .on("mu", |p| p.add(Linear::new(0)).add(PSpline::new(1)));
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
            Algorithm::NonCyclic => {
                if matches!(self.config.mstop, Mstop::PerParam(_)) {
                    return Err(BoostlssError::InvalidConfig(
                        "NonCyclic algorithm requires a Scalar Mstop".into(),
                    ));
                }
                crate::engine::noncyclical::fit_noncyclical(self, data)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scale {
    Link,
    Response,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStep {
    pub param_idx: usize,
    pub learner_idx: usize,
    pub update: LearnerUpdate,
    pub risk_reduction: f64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
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

        for update in &self.updates {
            if update.param_idx != k {
                continue;
            }

            let learner = &mut self.learners[update.learner_idx].1;
            match &update.update {
                LearnerUpdate::Linear(coef) => {
                    if let BaseLearner::RandomEffects(re) = learner {
                        let x_col = data.design().column(re.feature_idx).to_owned();
                        let mut u_hat = ndarray::Array1::zeros(x_col.len());
                        for (i, &val) in x_col.iter().enumerate() {
                            if val >= 0.0 && val.fract() == 0.0 {
                                let idx = val as usize;
                                if idx < coef.len() {
                                    u_hat[i] = coef[idx];
                                }
                            }
                        }
                        pred = pred + u_hat;
                    } else if let BaseLearner::BivariatePSpline(bp) = learner {
                        let design = bp.build_design(data)?;
                        let u_hat = design.dot(coef);
                        pred = pred + u_hat;
                    } else {
                        let design = learner.build_design(data)?;
                        let u_hat = design.dot(coef);
                        pred = pred + u_hat;
                    }
                }
                LearnerUpdate::Stump {
                    split_val,
                    left_val,
                    right_val,
                } => {
                    let feature_idx = if let BaseLearner::Stump(st) = learner {
                        st.feature_idx
                    } else {
                        0
                    };
                    let x_col = data.design().column(feature_idx).to_owned();
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

    pub fn feature_importance(&self) -> Vec<f64> {
        let mut importances = vec![0.0; self.learners.len()];
        for update in &self.updates {
            if update.learner_idx < importances.len() {
                importances[update.learner_idx] += update.risk_reduction;
            }
        }
        importances
    }

    pub fn partial_dependence(
        &mut self,
        data: &Dataset,
        param: &str,
        feature_idx: usize,
        grid: &[f64],
    ) -> Result<Vec<f64>, BoostlssError> {
        let mut results = Vec::with_capacity(grid.len());

        if feature_idx >= data.design().ncols() {
            return Err(BoostlssError::DataError(format!(
                "Feature index {} out of bounds for design matrix with {} columns",
                feature_idx,
                data.design().ncols()
            )));
        }

        for &val in grid {
            let mut modified_design = data.design().clone();
            modified_design.column_mut(feature_idx).fill(val);

            let modified_data = Dataset::new(
                modified_design,
                data.response().clone(),
                data.weights().cloned(),
            )?;

            let preds = self.predict(&modified_data, param, Scale::Link)?;
            let mean_pred = preds.sum() / (preds.len() as f64);
            results.push(mean_pred);
        }

        Ok(results)
    }
}

impl<F: Family + serde::Serialize + serde::de::DeserializeOwned> Fitted<F> {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BoostlssError> {
        let file =
            File::create(path).map_err(|e| BoostlssError::SerializationError(e.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)
            .map_err(|e| BoostlssError::SerializationError(e.to_string()))?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BoostlssError> {
        let file =
            File::open(path).map_err(|e| BoostlssError::SerializationError(e.to_string()))?;
        let reader = BufReader::new(file);
        let fitted: Self = serde_json::from_reader(reader)
            .map_err(|e| BoostlssError::SerializationError(e.to_string()))?;
        Ok(fitted)
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
            .on("mu", |p| p.add(Linear::new(0)))
            .unwrap();

        assert_eq!(model.learners().len(), 1);
        assert_eq!(model.learners()[0].0, 0);
    }

    #[test]
    fn test_boostlss_on_invalid_param() {
        let result =
            BoostLss::new(GaussianLss::new()).on("invalid_param", |p| p.add(Linear::new(0)));

        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }

    #[test]
    fn test_boostlss_on_multiple_learners() {
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| {
                p.add(Linear::new(0)).add(crate::learner::PSpline::new(1))
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
        let data =
            Dataset::new(Array2::<f64>::zeros((5, 2)), Array1::<f64>::zeros(5), None).unwrap();

        let pred = fitted.predict(&data, "mu", Scale::Link).unwrap();
        assert_eq!(pred.len(), 5);
        assert_eq!(pred, Array1::<f64>::zeros(5));
    }

    #[test]
    fn test_update_step_has_risk_reduction() {
        let update = UpdateStep {
            param_idx: 0,
            learner_idx: 1,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 1.5,
        };
        assert_eq!(update.risk_reduction, 1.5);
    }

    #[test]
    fn test_feature_importance() {
        let family = GaussianLss::new();
        let learners = vec![
            (0, BaseLearner::Linear(Linear::new(0))),
            (0, BaseLearner::Linear(Linear::new(1))),
        ];
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], learners);

        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 2.0,
        });
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 1,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 1.5,
        });
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(ndarray::Array1::zeros(2)),
            risk_reduction: 0.5,
        });

        let importance = fitted.feature_importance();
        assert_eq!(importance.len(), 2);
        assert_eq!(importance[0], 2.5); // 2.0 + 0.5
        assert_eq!(importance[1], 1.5);
    }

    #[test]
    fn test_partial_dependence() {
        use ndarray::{array, Array1, Array2};
        let family = GaussianLss::new();
        let learners = vec![(0, BaseLearner::Linear(Linear::new(0)))];
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], learners);

        // Mock an update where predicted mu = 2.0 * x
        fitted.updates.push(UpdateStep {
            param_idx: 0,
            learner_idx: 0,
            update: crate::learner::LearnerUpdate::Linear(array![0.0, 2.0]),
            risk_reduction: 0.0,
        });

        let data =
            Dataset::new(Array2::<f64>::zeros((5, 2)), Array1::<f64>::zeros(5), None).unwrap();
        let grid = vec![1.0, 2.0, 3.0];

        // We evaluate feature_idx 1 (the 'x' column in the design matrix, since 0 is intercept for Linear usually, but let's test feature_idx=0 here)
        let pd = fitted.partial_dependence(&data, "mu", 0, &grid).unwrap();

        assert_eq!(pd.len(), 3);
        // pred = offset(0) + 0.0(intercept) + 2.0 * grid_val
        assert_eq!(pd[0], 2.0); // 2.0 * 1.0
        assert_eq!(pd[1], 4.0); // 2.0 * 2.0
        assert_eq!(pd[2], 6.0); // 2.0 * 3.0
    }

    #[test]
    fn test_partial_dependence_out_of_bounds() {
        use ndarray::{Array1, Array2};
        let family = GaussianLss::new();
        let mut fitted = Fitted::new(family, vec![0.0, 0.0], vec![]);
        let data =
            Dataset::new(Array2::<f64>::zeros((5, 2)), Array1::<f64>::zeros(5), None).unwrap();
        let grid = vec![1.0, 2.0];

        let result = fitted.partial_dependence(&data, "mu", 999, &grid);
        assert!(matches!(result, Err(BoostlssError::DataError(_))));
    }

    #[test]
    fn test_predict_random_effects_out_of_bounds() {
        use crate::engine::Mstop;
        use crate::learner::RandomEffects;
        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(RandomEffects::new(0)))
            .unwrap()
            .step_length(0.1)
            .mstop(Mstop::Scalar(1));

        let x_train = ndarray::array![0.0, 1.0, 0.0, 1.0];
        let y_train = ndarray::array![10.0, 20.0, 10.0, 20.0];
        let ds_train = Dataset::new(
            ndarray::Array2::from_shape_vec((4, 1), x_train.to_vec()).unwrap(),
            y_train,
            None,
        )
        .unwrap();

        let mut fitted = model.fit(&ds_train).unwrap();

        // Predict on unseen group index (2.0) and negative index (-1.0)
        let x_test = ndarray::array![0.0, 2.0, -1.0];
        let ds_test = Dataset::new(
            ndarray::Array2::from_shape_vec((3, 1), x_test.to_vec()).unwrap(),
            ndarray::Array1::zeros(3),
            None,
        )
        .unwrap();

        let preds = fitted.predict(&ds_test, "mu", Scale::Link).unwrap();

        // Unseen/invalid groups should get exactly 0.0 addition to the intercept
        // First element is seen, so it has a non-intercept effect.
        // Elements 1 and 2 are unseen, so they should equal the global offset exactly.
        // using fitted.offsets[0] instead of fitted.offset(0) since offset() is not defined
        let offset = fitted.offsets[0];
        assert!((preds[0] - offset).abs() > 1e-10);
        assert!((preds[1] - offset).abs() < 1e-10);
        assert!((preds[2] - offset).abs() < 1e-10);
    }

    #[test]
    fn test_boostlss_fit_noncyclic_requires_scalar_mstop() {
        use crate::data::Dataset;
        use crate::engine::Mstop;
        use crate::family::GaussianLss;
        use crate::learner::Linear;
        use ndarray::{array, Array2};

        let x = Array2::<f64>::zeros((2, 1));
        let y = array![1.0, 2.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new(0)))
            .unwrap()
            .algorithm(crate::engine::Algorithm::NonCyclic)
            .mstop(Mstop::PerParam(vec![10, 10])); // Invalid for NonCyclic

        let result = model.fit(&data);
        assert!(matches!(result, Err(BoostlssError::InvalidConfig(_))));
    }
}
