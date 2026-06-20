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
}
