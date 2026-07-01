use crate::data::Dataset;
use crate::engine::stabilization::stabilize;
use crate::engine::Mstop;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::learner::{LearnerFit, LearnerUpdate};
use crate::model::{BoostLss, Fitted, UpdateStep};

struct CachedLearner {
    param_idx: usize,
    learner_idx: usize,
    fit_state: LearnerFit,
}

pub fn fit_noncyclical<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
    eval_data: Option<&Dataset>,
    early_stopping_rounds: Option<usize>,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();

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
        Mstop::PerParam(_) => {
            return Err(BoostlssError::InvalidConfig(
                "NonCyclic algorithm requires a Scalar Mstop".into(),
            ));
        }
    };
    let nu = config.step_length;

    let mut updates = Vec::new();

    for _m in 1..=max_mstop {
        let base_nll = family.nll(data, &current_predictions)?;
        let num_params = family.params().len();

        let mut candidate_updates = Vec::new();

        // Stage 1: Find best learner per parameter using RSS on negative gradients
        for k in 0..num_params {
            let mut gradients = family.ngradient(data, &current_predictions, k)?;
            stabilize(&mut gradients, config.stabilization, data.weights());

            let mut best_rss = f64::INFINITY;
            let mut best_update: Option<LearnerUpdate> = None;
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
                candidate_updates.push((k, l_idx, update, u_hat));
            }
        }

        // Stage 2: Select the candidate that minimizes total NLL
        let mut best_nll = f64::INFINITY;
        let mut selected_candidate = None;

        for (k, l_idx, update, u_hat) in candidate_updates {
            // Apply step length nu
            let step = &u_hat * nu;

            let original_predictions_k = current_predictions[k].clone();

            // Temporarily apply update
            current_predictions[k] = &original_predictions_k + &step;

            let nll = family.nll(data, &current_predictions)?;

            if nll < best_nll {
                best_nll = nll;
                selected_candidate = Some((k, l_idx, update, step));
            }

            // Restore original predictions
            current_predictions[k] = original_predictions_k;
        }

        if let Some((k, l_idx, update, step)) = selected_candidate {
            current_predictions[k] = &current_predictions[k] + &step;
            let risk_reduction = base_nll - best_nll;

            // Standard boosting applies the update even if empirical NLL increases slightly,
            // but we use max(0.0) for risk_reduction to avoid negative importance.
            let risk_reduction = risk_reduction.max(0.0);

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
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    fitted.eval_results = crate::model::EvalResults {
        train_loss: vec![],
        val_loss: None,
    };
    fitted.best_iteration = max_mstop;
    Ok(fitted)
}

pub fn fit_noncyclical_outer<F: Family + Clone>(
    model: BoostLss<F>,
    data: &Dataset,
    eval_data: Option<&Dataset>,
    early_stopping_rounds: Option<usize>,
) -> Result<Fitted<F>, BoostlssError> {
    let mut current_predictions = Vec::new();
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
        Mstop::PerParam(_) => {
            return Err(BoostlssError::InvalidConfig(
                "NonCyclic algorithm requires a Scalar Mstop".into(),
            ));
        }
    };
    let nu = config.step_length;
    let mut updates = Vec::new();

    for _m in 1..=max_mstop {
        let base_nll = family.nll(data, &current_predictions)?;
        let num_params = family.params().len();

        let mut best_nll = f64::INFINITY;
        let mut best_candidate = None;

        for k in 0..num_params {
            let mut gradients = family.ngradient(data, &current_predictions, k)?;
            stabilize(&mut gradients, config.stabilization, data.weights());

            let original_predictions_k = current_predictions[k].clone();

            for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                let update = cached
                    .fit_state
                    .fit_update(gradients.view(), data.weights().map(|w| w.view()));
                let u_hat = cached.fit_state.predict_update(&update, data);
                let step = &u_hat * nu;

                // Temporarily apply update
                current_predictions[k] = &original_predictions_k + &step;
                let nll = family.nll(data, &current_predictions)?;

                if nll < best_nll {
                    best_nll = nll;
                    best_candidate = Some((k, cached.learner_idx, update.clone(), step.clone()));
                }
            }

            // Restore original predictions
            current_predictions[k] = original_predictions_k;
        }

        if let Some((k, l_idx, mut update, step)) = best_candidate {
            current_predictions[k] = &current_predictions[k] + &step;
            let risk_reduction = (base_nll - best_nll).max(0.0);
            update.scale(nu);

            updates.push(UpdateStep {
                param_idx: k,
                learner_idx: l_idx,
                risk_reduction,
                update,
            });
        }
    }

    let mut fitted = Fitted::new(family, offsets, learners);
    fitted.updates = updates;
    fitted.eval_results = crate::model::EvalResults {
        train_loss: vec![],
        val_loss: None,
    };
    fitted.best_iteration = max_mstop;
    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::engine::{Algorithm, Mstop};
    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use ndarray::array;

    #[test]
    fn test_fit_noncyclical_selects_best_param() {
        // x is perfectly correlated with y
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new(0)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new(0)))
            .unwrap()
            .algorithm(Algorithm::NonCyclic)
            .mstop(Mstop::Scalar(1));

        let fitted = fit_noncyclical(model, &data, None, None).unwrap();

        // There should be exactly 1 update since mstop=1
        assert_eq!(fitted.updates.len(), 1);

        // Since mu is far off (mean vs true values), it should be updated to reduce NLL.
        // The single update should target param_idx 0 (mu).
        assert_eq!(fitted.updates[0].param_idx, 0);
    }

    #[test]
    fn test_algorithm_variants_exist() {
        use crate::engine::Algorithm;
        let _a1 = Algorithm::Cyclic;
        let _a2 = Algorithm::NonCyclic;
        let _a3 = Algorithm::NonCyclicOuter;
    }

    #[test]
    fn test_fit_noncyclical_outer() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new(0)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new(0)))
            .unwrap()
            .algorithm(Algorithm::NonCyclicOuter)
            .mstop(Mstop::Scalar(1));

        // fit_noncyclical_outer should only generate 1 update and it should reduce NLL
        let fitted = fit_noncyclical_outer(model, &data, None, None).unwrap();
        assert_eq!(fitted.updates.len(), 1);
        assert!(fitted.updates[0].risk_reduction > 0.0);
    }
}
