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
    eval_data: Option<&Dataset>,
    early_stopping_rounds: Option<usize>,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();

    // 1. Initialize offsets
    let offsets = model.family().init_offsets(data)?;

    for offset in &offsets {
        current_predictions.push(ndarray::Array1::from_elem(data.n_obs(), *offset));
    }

    let mut cached_learners = Vec::new();

    let (family, config, mut learners) = model.into_parts();
    for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
        let fit_state = learner.initialize(data)?;
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
    let mut updates_per_iteration = Vec::new();
    let mut best_val_nll = f64::INFINITY;
    let mut best_iteration = 0;
    let mut train_losses = Vec::new();
    let mut val_losses = if eval_data.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    let mut current_eval_predictions = if let Some(e_data) = eval_data {
        let mut preds = Vec::new();
        for offset in &offsets {
            preds.push(ndarray::Array1::from_elem(e_data.n_obs(), *offset));
        }
        Some(preds)
    } else {
        None
    };

    for m in 1..=max_mstop {
        let mut updates_in_m = 0;
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

            let base_rss = match data.weights() {
                Some(w) => (&gradients * &gradients * w).sum(),
                None => (&gradients * &gradients).sum(),
            };

            let mut best_rss = f64::INFINITY;
            let mut best_update: Option<crate::learner::LearnerUpdate> = None;
            let mut best_u_hat: Option<ndarray::Array1<f64>> = None;
            let mut best_learner_idx = None;

            for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                let update = cached
                    .fit_state
                    .fit_update(gradients.view(), data.weights().map(|w| w.view()));

                let u_hat = cached.fit_state.predict_update(&update, data);

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
                let risk_reduction = base_rss - best_rss;
                updates.push(UpdateStep {
                    param_idx: k,
                    learner_idx: l_idx,
                    risk_reduction,
                    update: {
                        let mut u = update.clone();
                        u.scale(nu);
                        u
                    },
                });
                updates_in_m += 1;

                if let Some(ref mut eval_preds) = current_eval_predictions {
                    if let Some(e_data) = eval_data {
                        let p = cached_learners
                            .iter()
                            .find(|c| c.learner_idx == l_idx)
                            .unwrap()
                            .fit_state
                            .predict_update(&update, e_data);
                        eval_preds[k] = &eval_preds[k] + &(&p * nu);
                    }
                }
            }
        }

        updates_per_iteration.push(updates_in_m);

        let train_nll = family.nll(data, &current_predictions)?;
        train_losses.push(train_nll);

        if let Some(ref eval_preds) = current_eval_predictions {
            if let Some(e_data) = eval_data {
                let val_nll = family.nll(e_data, eval_preds)?;
                val_losses.as_mut().unwrap().push(val_nll);

                if val_nll < best_val_nll {
                    best_val_nll = val_nll;
                    best_iteration = m;
                }

                if let Some(patience) = early_stopping_rounds {
                    if m - best_iteration >= patience {
                        break;
                    }
                }
            }
        } else {
            best_iteration = m;
        }
    }

    if early_stopping_rounds.is_some() && eval_data.is_some() {
        let keep_count: usize = updates_per_iteration.iter().take(best_iteration).sum();
        updates.truncate(keep_count);
        train_losses.truncate(best_iteration);
        if let Some(ref mut vl) = val_losses {
            vl.truncate(best_iteration);
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    fitted.eval_results = crate::model::EvalResults {
        train_loss: train_losses,
        val_loss: val_losses,
    };
    fitted.best_iteration = best_iteration;
    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use crate::model::Scale;
    use ndarray::array;

    #[test]
    fn test_fit_cyclical() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0]; // Perfect linear relationship
        let data = Dataset::new(x, y.clone(), None, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new(0).intercept(true)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new(0).intercept(true)))
            .unwrap()
            .algorithm(crate::engine::Algorithm::Cyclic)
            .mstop(Mstop::PerParam(vec![2, 2]));

        let mut fitted = fit_cyclical(model, &data, None, None).unwrap();

        let pred_mu = fitted.predict(&data, "mu", Scale::Response).unwrap();
        // Since it's a perfect relationship, predictions should move towards y
        assert!(pred_mu[3] > pred_mu[0]); // monotonic
    }

    #[test]
    fn test_risk_reduction_calculation() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new(0)))
            .unwrap()
            .algorithm(crate::engine::Algorithm::Cyclic)
            .mstop(Mstop::Scalar(1));

        let fitted = fit_cyclical(model, &data, None, None).unwrap();

        assert_eq!(fitted.updates.len(), 1);
        assert!(fitted.updates[0].risk_reduction > 0.0);
    }

    #[test]
    fn test_early_stopping_cyclical() {
        use crate::model::BoostLss;
        use ndarray::{Array1, Array2};

        let n = 100;
        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[[i, 0]] = (i as f64) / (n as f64);
            y[i] = x[[i, 0]] * 2.0;
        }
        let data = Dataset::new(x.clone(), y.clone(), None, None).unwrap();

        let family = crate::family::GaussianLss::new();
        // High mstop, should stop early
        let model = BoostLss::new(family)
            .mstop(crate::engine::Mstop::Scalar(1000))
            .step_length(0.1)
            .algorithm(crate::engine::Algorithm::Cyclic)
            .on("mu", |p| {
                p.add(crate::learner::Linear::new(0).intercept(true))
            })
            .unwrap();

        // Use identical data for eval, just to test tracking/stopping mechanics
        let fitted = model.fit(&data, Some(&data), Some(5)).unwrap();

        assert!(fitted.best_iteration < 1000);
        assert_eq!(fitted.updates.len(), fitted.best_iteration); // 1 param (sigma is not fitted in this test config to keep it simple)
        assert!(fitted.eval_results.val_loss.is_some());
        assert_eq!(fitted.eval_results.train_loss.len(), fitted.best_iteration);
    }
}
