use crate::engine::Mstop;
use ndarray::Array1;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub enum Resampling {
    Bootstrap { b: usize },
    KFold { k: usize },
    Subsampling { b: usize, prob: f64 },
}

impl Resampling {
    pub fn generate_weights(&self, n: usize, rng: &mut impl Rng) -> Vec<Array1<f64>> {
        match self {
            Resampling::Bootstrap { b } => {
                let mut weights = Vec::with_capacity(*b);
                for _ in 0..*b {
                    let mut w = Array1::zeros(n);
                    for _ in 0..n {
                        let idx = rng.gen_range(0..n);
                        w[idx] += 1.0;
                    }
                    weights.push(w);
                }
                weights
            }
            Resampling::KFold { k } => {
                assert!(*k > 0, "KFold: k must be greater than 0");
                let mut weights = Vec::with_capacity(*k);
                let fold_size = n / k;
                let remainder = n % k;
                let mut start = 0;
                for i in 0..*k {
                    let current_fold_size = if i < remainder {
                        fold_size + 1
                    } else {
                        fold_size
                    };
                    let end = start + current_fold_size;

                    let mut w = Array1::ones(n);
                    for j in start..end {
                        w[j] = 0.0;
                    }
                    weights.push(w);
                    start = end;
                }
                weights
            }
            Resampling::Subsampling { b, prob } => {
                assert!(
                    *prob >= 0.0 && *prob <= 1.0,
                    "Subsampling: prob must be in [0.0, 1.0]"
                );
                let mut weights = Vec::with_capacity(*b);
                let num_samples = (*prob * n as f64).round() as usize;
                for _ in 0..*b {
                    let mut w = Array1::zeros(n);
                    let indices = rand::seq::index::sample(rng, n, num_samples);
                    for idx in indices.into_iter() {
                        w[idx] = 1.0;
                    }
                    weights.push(w);
                }
                weights
            }
        }
    }
}

pub fn make_grid(params_count: usize, mstop_max: usize, length_out: usize) -> Vec<Mstop> {
    if params_count == 0 || length_out == 0 || mstop_max == 0 {
        return vec![];
    }

    let mut vals = Vec::with_capacity(length_out);
    if length_out == 1 {
        vals.push(mstop_max);
    } else {
        let ln_start = 0.0f64; // ln(1)
        let ln_end = (mstop_max as f64).ln();
        for i in 0..length_out {
            let log_val = ln_start + (ln_end - ln_start) * (i as f64) / ((length_out - 1) as f64);
            let val = log_val.exp().round() as usize;
            let val = val.max(1).min(mstop_max);
            vals.push(val);
        }
    }
    vals.dedup();

    let mut grid = Vec::new();
    let mut current = vec![0; params_count];

    loop {
        let config: Vec<usize> = current.iter().map(|&idx| vals[idx]).collect();
        grid.push(Mstop::PerParam(config));

        let mut i = 0;
        while i < params_count {
            current[i] += 1;
            if current[i] < vals.len() {
                break;
            }
            current[i] = 0;
            i += 1;
        }
        if i == params_count {
            break;
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kfold_weights() {
        let resampling = Resampling::KFold { k: 2 };
        let mut rng = StdRng::seed_from_u64(42);
        let weights = resampling.generate_weights(10, &mut rng);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].sum(), 5.0);
        assert_eq!(weights[1].sum(), 5.0);
    }

    #[test]
    fn test_make_grid() {
        let grid = make_grid(2, 10, 3);
        // length_out = 3, min = 1, max = 10. log-spaced rounded: 1, 3, 10.
        // grid size = 3 * 3 = 9
        assert_eq!(grid.len(), 9);
        assert!(matches!(&grid[0], Mstop::PerParam(v) if v == &vec![1, 1]));
    }
}
