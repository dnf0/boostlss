use crate::error::BoostlssError;

#[derive(Clone, Debug, PartialEq)]
pub enum StabselMode {
    Joint,
    Independent,
}

#[derive(Clone, Debug)]
pub struct StabselConfig {
    pub b: usize,
    pub pfer: Option<f64>,
    pub pi_thr: Option<f64>,
    pub q: Option<usize>,
    pub mode: StabselMode,
    pub p: usize,
}

impl StabselConfig {
    pub fn new(
        b: usize,
        pfer: Option<f64>,
        pi_thr: Option<f64>,
        q: Option<usize>,
        mode: StabselMode,
        p: usize,
    ) -> Result<Self, BoostlssError> {
        if p == 0 {
            return Err(BoostlssError::InvalidStabselConfig(
                "p must be greater than 0".to_string(),
            ));
        }

        if let Some(q_val) = q {
            if q_val == 0 || q_val > p {
                return Err(BoostlssError::InvalidStabselConfig(
                    "q must be in 1..=p".to_string(),
                ));
            }
        }

        let provided = pfer.is_some() as u8 + pi_thr.is_some() as u8 + q.is_some() as u8;

        if provided != 2 {
            return Err(BoostlssError::InvalidStabselConfig(
                "Exactly two of (pfer, pi_thr, q) must be provided".to_string(),
            ));
        }

        let mut config = Self {
            b,
            pfer,
            pi_thr,
            q,
            mode,
            p,
        };

        config.resolve_bounds()?;

        if let Some(q_val) = config.q {
            if q_val == 0 || q_val > p {
                return Err(BoostlssError::InvalidStabselConfig(
                    "Derived q must be in 1..=p".to_string(),
                ));
            }
        }

        Ok(config)
    }

    fn resolve_bounds(&mut self) -> Result<(), BoostlssError> {
        // Shah & Samworth (2013) bounds: PFER <= q^2 / ((2 * pi_thr - 1) * p)
        match (self.pfer, self.pi_thr, self.q) {
            (None, Some(pi_thr), Some(q)) => {
                if pi_thr <= 0.5 || pi_thr >= 1.0 {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "pi_thr must be in (0.5, 1.0)".to_string(),
                    ));
                }
                self.pfer = Some((q as f64 * q as f64) / ((2.0 * pi_thr - 1.0) * self.p as f64));
            }
            (Some(pfer), None, Some(q)) => {
                if pfer <= 0.0 {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "pfer must be > 0.0".to_string(),
                    ));
                }
                let pi_thr = ((q as f64 * q as f64) / (pfer * self.p as f64) + 1.0) / 2.0;
                if pi_thr <= 0.5 || pi_thr >= 1.0 {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "Derived pi_thr must be in (0.5, 1.0). Adjust q or pfer.".to_string(),
                    ));
                }
                self.pi_thr = Some(pi_thr);
            }
            (Some(pfer), Some(pi_thr), None) => {
                if pi_thr <= 0.5 || pi_thr >= 1.0 || pfer <= 0.0 {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "Invalid pi_thr or pfer".to_string(),
                    ));
                }
                let q_f64 = (pfer * (2.0 * pi_thr - 1.0) * self.p as f64).sqrt();
                let q_val = q_f64.floor() as usize;
                if q_val == 0 {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "Derived q is 0. Adjust pi_thr or pfer.".to_string(),
                    ));
                }
                self.q = Some(q_val);
            }
            _ => unreachable!("Config validated to have exactly 2 of 3 parameters"),
        }
        Ok(())
    }
}

pub struct StabselResult {
    pub frequencies: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
    pub q: usize,
    pub pfer: f64,
    pub pi_thr: f64,
    pub b: usize,
}

struct CachedLearner {
    param_idx: usize,
    learner_idx: usize,
    fit_state: crate::learner::LearnerFit,
}

pub fn run_stabsel<F: crate::family::Family + Clone + Send + Sync>(
    model: &crate::model::BoostLss<F>,
    data: &crate::data::Dataset,
    mstop: crate::engine::Mstop,
    config: &StabselConfig,
) -> Result<StabselResult, BoostlssError> {
    use ndarray::Array1;
    use rand::{Rng, SeedableRng};
    use std::collections::{HashMap, HashSet};

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let b_runs = config.b;
    let n = data.n_obs();
    let num_samples = n / 2;
    let q_limit = config.q.unwrap();

    let mut seeds = Vec::with_capacity(b_runs);
    let mut rng = rand::thread_rng();
    for _ in 0..b_runs {
        seeds.push(rng.gen::<u64>());
    }

    #[cfg(feature = "parallel")]
    let iter = seeds.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = seeds.iter();

    // Each thread returns the active set of base learners (Param -> HashSet<LearnerName>)
    let results: Result<Vec<HashMap<String, HashSet<String>>>, BoostlssError> = iter
        .map(|&seed| {
            let mut run_rng = rand::rngs::StdRng::seed_from_u64(seed);

            // Subsample weights
            let mut w = Array1::zeros(n);
            let indices = rand::seq::index::sample(&mut run_rng, n, num_samples);
            for idx in indices.into_iter() {
                w[idx] = 1.0;
            }

            let mut run_data = data.clone();
            run_data.set_weights(w)?;

            let (family, model_config, mut learners) = model.clone().into_parts();

            let mut current_predictions = Vec::new();
            let offsets = family.init_offsets(&run_data)?;
            for offset in &offsets {
                current_predictions.push(Array1::from_elem(n, *offset));
            }

            let mut cached_learners = Vec::new();
            for (idx, (param_idx, learner)) in learners.iter_mut().enumerate() {
                let fit_state = learner.initialize(&run_data)?;
                cached_learners.push(CachedLearner {
                    param_idx: *param_idx,
                    learner_idx: idx,
                    fit_state,
                });
            }

            let max_mstop = match mstop {
                crate::engine::Mstop::Scalar(m) => m,
                crate::engine::Mstop::PerParam(_) => {
                    return Err(BoostlssError::InvalidStabselConfig(
                        "Mstop::Vector not supported in stabsel".to_string(),
                    ))
                }
            };
            let nu = model_config.step_length;

            let params = family.params();
            let mut active_sets: HashMap<String, HashSet<String>> = HashMap::new();
            for p in params {
                active_sets.insert(p.name.clone(), HashSet::new());
            }

            let mut global_active_count = 0;

            'outer: for _m in 1..=max_mstop {
                for k in 0..params.len() {
                    let mut gradients = family.ngradient(&run_data, &current_predictions, k)?;

                    crate::engine::stabilization::stabilize(
                        &mut gradients,
                        model_config.stabilization,
                        run_data.weights(),
                    );

                    let mut best_rss = f64::INFINITY;
                    let mut best_update: Option<crate::learner::LearnerUpdate> = None;
                    let mut best_u_hat: Option<Array1<f64>> = None;
                    let mut best_learner_idx = None;

                    for cached in cached_learners.iter().filter(|c| c.param_idx == k) {
                        let update = cached
                            .fit_state
                            .fit_update(gradients.view(), run_data.weights().map(|w| w.view()));

                        let u_hat = cached.fit_state.predict_update(&update, &run_data);

                        let residuals = &gradients - &u_hat;
                        let rss = match run_data.weights() {
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

                    if let (Some(_update), Some(u_hat), Some(l_idx)) =
                        (best_update, best_u_hat, best_learner_idx)
                    {
                        current_predictions[k] = &current_predictions[k] + &(&u_hat * nu);

                        let param_name = &params[k].name;
                        let learner_name = learners[l_idx].1.name();

                        let set = active_sets.get_mut(param_name).unwrap();
                        if set.insert(learner_name) {
                            global_active_count += 1;
                        }

                        // Early stopping
                        if config.mode == StabselMode::Joint && global_active_count >= q_limit {
                            break 'outer;
                        }

                        if config.mode == StabselMode::Independent {
                            let all_reached = active_sets.values().all(|s| s.len() >= q_limit);
                            if all_reached {
                                break 'outer;
                            }
                        }
                    }
                }
            }

            Ok(active_sets)
        })
        .collect();

    let results = results?;

    // Aggregate frequencies
    let mut frequencies: std::collections::HashMap<
        String,
        std::collections::HashMap<String, usize>,
    > = std::collections::HashMap::new();
    let params = model.param_names();
    for p in params {
        frequencies.insert(p, std::collections::HashMap::new());
    }

    for run_active in results {
        for (param, learners) in run_active {
            let param_freq = frequencies.get_mut(&param).unwrap();
            for learner in learners {
                *param_freq.entry(learner).or_insert(0) += 1;
            }
        }
    }

    Ok(StabselResult {
        frequencies,
        q: config.q.unwrap(),
        pfer: config.pfer.unwrap(),
        pi_thr: config.pi_thr.unwrap(),
        b: b_runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_config_not_two_params() {
        let err = StabselConfig::new(100, Some(1.0), Some(0.6), Some(10), StabselMode::Joint, 100)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "Exactly two of (pfer, pi_thr, q) must be provided");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_resolve_pfer() {
        let config =
            StabselConfig::new(100, None, Some(0.6), Some(10), StabselMode::Joint, 100).unwrap();
        assert_eq!(config.q.unwrap(), 10);
        assert_eq!(config.pi_thr.unwrap(), 0.6);
        assert!((config.pfer.unwrap() - 5.0).abs() < 1e-6); // 100 / (0.2 * 100) = 100 / 20 = 5.0
    }

    #[test]
    fn test_resolve_pi_thr() {
        let config =
            StabselConfig::new(100, Some(5.0), None, Some(10), StabselMode::Joint, 100).unwrap();
        assert_eq!(config.q.unwrap(), 10);
        assert_eq!(config.pfer.unwrap(), 5.0);
        assert!((config.pi_thr.unwrap() - 0.6).abs() < 1e-6); // (100 / (5 * 100) + 1) / 2 = (1/5 + 1)/2 = 0.6
    }

    #[test]
    fn test_resolve_q() {
        let config =
            StabselConfig::new(100, Some(5.0), Some(0.7), None, StabselMode::Joint, 100).unwrap();
        assert_eq!(config.pfer.unwrap(), 5.0);
        assert_eq!(config.pi_thr.unwrap(), 0.7);
        assert_eq!(config.q.unwrap(), 14); // sqrt(5 * 0.4 * 100) = sqrt(200) = 14.14 -> floor is 14
    }

    #[test]
    fn test_invalid_pi_thr() {
        let err = StabselConfig::new(100, None, Some(0.4), Some(10), StabselMode::Joint, 100)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "pi_thr must be in (0.5, 1.0)");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_invalid_config_p_zero() {
        let err =
            StabselConfig::new(100, None, Some(0.6), Some(10), StabselMode::Joint, 0).unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "p must be greater than 0");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_invalid_config_q_zero() {
        let err =
            StabselConfig::new(100, None, Some(0.6), Some(0), StabselMode::Joint, 100).unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "q must be in 1..=p");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_invalid_config_q_greater_than_p() {
        let err = StabselConfig::new(100, None, Some(0.6), Some(110), StabselMode::Joint, 100)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "q must be in 1..=p");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }

    #[test]
    fn test_invalid_config_derived_q_greater_than_p() {
        let err = StabselConfig::new(100, Some(61.0), Some(0.6), None, StabselMode::Joint, 10)
            .unwrap_err();
        match err {
            BoostlssError::InvalidStabselConfig(msg) => {
                assert_eq!(msg, "Derived q must be in 1..=p");
            }
            _ => panic!("Expected InvalidStabselConfig"),
        }
    }
}
