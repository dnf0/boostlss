use crate::data::Dataset;
use crate::engine::stabilization::stabilize;
use crate::engine::Mstop;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::LearnerFit;
use crate::model::{BoostLss, Fitted, UpdateStep};

struct CachedLearner {
    param_idx: usize,
    learner_idx: usize,
    fit_state: LearnerFit,
}

pub fn fit_cyclical<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();

    // 1. Initialize offsets
    let offsets = model.family().init_offsets(data)?;

    for offset in &offsets {
        current_predictions.push(ndarray::Array1::from_elem(data.n_obs(), *offset));
    }

    let mut cached_learners = Vec::new();
    let x_col = data.design().column(0).to_owned();

    let (family, config, mut learners) = model.into_parts();
    for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
        let fit_state = learner.initialize(&x_col)?;
        cached_learners.push(CachedLearner {
            param_idx: *param_idx,
            learner_idx: idx,
            fit_state,
        });
    }

    let max_mstop = match config.mstop {
        Mstop::Scalar(m) => m,
        Mstop::PerParam(ref v) => *v.iter().max().unwrap_or(&0),
    };
    let nu = config.step_length;

    let mut updates = Vec::new();

    for m in 1..=max_mstop {
        for k in 0..family.params().len() {
            let mstop_k = match config.mstop {
                Mstop::Scalar(ms) => ms,
                Mstop::PerParam(ref v) => v[k],
            };
            if m > mstop_k {
                continue;
            }

            let mut gradients = family.ngradient(data, &current_predictions, k)?;

            stabilize(&mut gradients, config.stabilization, data.weights());

            let mut best_rss = f64::INFINITY;
            let mut best_update: Option<crate::learner::LearnerUpdate> = None;
            let mut best_u_hat: Option<ndarray::Array1<f64>> = None;
            let mut best_learner_idx = None;

            for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                let update = cached
                    .fit_state
                    .fit_update(gradients.view(), data.weights().map(|w| w.view()));

                let u_hat = match &update {
                    crate::learner::LearnerUpdate::Linear(coef) => {
                        if let LearnerFit::Linear(state) = &cached.fit_state {
                            state.design.dot(coef)
                        } else {
                            unreachable!()
                        }
                    }
                    crate::learner::LearnerUpdate::Stump {
                        split_val,
                        left_val,
                        right_val,
                    } => {
                        let x_col = data.design().column(0);
                        x_col.mapv(|val| {
                            if val <= *split_val {
                                *left_val
                            } else {
                                *right_val
                            }
                        })
                    }
                };

                let residuals = &gradients - &u_hat;
                let rss = match data.weights() {
                    Some(w) => (&residuals * &residuals * w).sum(),
                    None => (&residuals * &residuals).sum(),
                };

                if rss < best_rss {
                    best_rss = rss;
                    best_update = Some(update);
                    best_u_hat = Some(u_hat);
                    best_learner_idx = Some(cached.learner_idx);
                }
            }

            if let (Some(update), Some(u_hat), Some(l_idx)) =
                (best_update, best_u_hat, best_learner_idx)
            {
                current_predictions[k] = &current_predictions[k] + &(&u_hat * nu);
                updates.push(UpdateStep {
                    param_idx: k,
                    learner_idx: l_idx,
                    update: match update {
                        crate::learner::LearnerUpdate::Linear(coef) => {
                            crate::learner::LearnerUpdate::Linear(coef * nu)
                        }
                        crate::learner::LearnerUpdate::Stump {
                            split_val,
                            left_val,
                            right_val,
                        } => crate::learner::LearnerUpdate::Stump {
                            split_val,
                            left_val: left_val * nu,
                            right_val: right_val * nu,
                        },
                    },
                });
            }
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::family::GaussianLss;
    use crate::learner::{BaseLearner, Linear};
    use crate::model::Scale;
    use ndarray::array;

    #[test]
    fn test_fit_cyclical() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0]; // Perfect linear relationship
        let data = Dataset::new(x, y.clone(), None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .step_length(0.1)
            .mstop(crate::engine::Mstop::Scalar(10))
            .on("mu", BaseLearner::Linear(Linear::new("x").intercept(true)))
            .unwrap()
            .on(
                "sigma",
                BaseLearner::Linear(Linear::new("x").intercept(true)),
            )
            .unwrap();

        let mut fitted = fit_cyclical(model, &data).unwrap();

        let pred_mu = fitted.predict(&data, "mu", Scale::Response).unwrap();
        // Since it's a perfect relationship, predictions should move towards y
        assert!(pred_mu[3] > pred_mu[0]); // monotonic
    }
}
