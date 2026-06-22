use crate::data::Dataset;
use crate::engine::Mstop;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::model::BoostLss;
use crate::model::Scale;
use ndarray::Array1;
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

#[derive(Clone, Debug)]
pub struct CvRiskResult {
    pub risk_matrix: Vec<Vec<f64>>, // [fold][mstop_index]
    pub mean_risk: Vec<f64>,
    pub optimal_mstop: Mstop,
    pub mstop_grid: Vec<Mstop>,
}

#[cfg(feature = "parallel")]
pub trait FamilyBound: Family + Clone + Sync + Send {}
#[cfg(feature = "parallel")]
impl<T: Family + Clone + Sync + Send> FamilyBound for T {}

#[cfg(not(feature = "parallel"))]
pub trait FamilyBound: Family + Clone {}
#[cfg(not(feature = "parallel"))]
impl<T: Family + Clone> FamilyBound for T {}

pub struct CvRisk<F: FamilyBound> {
    model: BoostLss<F>,
    resampling: Resampling,
    mstop_max: usize,
    length_out: usize,
}

impl<F: FamilyBound> CvRisk<F> {
    pub fn new(model: BoostLss<F>, resampling: Resampling) -> Self {
        let mstop_max = match &model.config().mstop {
            Mstop::Scalar(m) => *m,
            Mstop::PerParam(ms) => *ms.iter().max().unwrap_or(&100),
        };
        Self {
            model,
            resampling,
            mstop_max,
            length_out: 10,
        }
    }

    pub fn grid_resolution(mut self, length_out: usize) -> Self {
        self.length_out = length_out;
        self
    }

    pub fn mstop_max(mut self, max: usize) -> Self {
        self.mstop_max = max;
        self
    }

    pub fn run(&self, data: &Dataset) -> Result<CvRiskResult, BoostlssError> {
        let params_count = self.model.family().params().len();
        let grid = make_grid(params_count, self.mstop_max, self.length_out);

        let mut rng = rand::thread_rng();
        let weights = self.resampling.generate_weights(data.n_obs(), &mut rng);

        #[cfg(feature = "parallel")]
        let weights_iter = weights.par_iter().enumerate();
        #[cfg(not(feature = "parallel"))]
        let weights_iter = weights.iter().enumerate();

        let risks: Result<Vec<Vec<f64>>, BoostlssError> = weights_iter
            .map(|(_fold_idx, w)| {
                let valid_indices: Vec<usize> = w
                    .iter()
                    .enumerate()
                    .filter(|(_, &wi)| wi == 0.0)
                    .map(|(i, _)| i)
                    .collect();

                let train_data = data.with_weights(w.clone())?;
                let valid_data = data.subset(&valid_indices)?;

                let mut fold_risks = vec![0.0; grid.len()];

                for (m_idx, m) in grid.iter().enumerate() {
                    let model = self.model.clone().mstop(m.clone());
                    let mut fitted = model.fit(&train_data)?;

                    let mut eta = Vec::with_capacity(params_count);
                    for param in self.model.family().params() {
                        let pred = fitted.predict(&valid_data, &param.name, Scale::Link)?;
                        eta.push(pred);
                    }

                    let risk =
                        self.model.family().nll(&valid_data, &eta)? / valid_data.n_obs() as f64;
                    fold_risks[m_idx] = risk;
                }

                Ok(fold_risks)
            })
            .collect();

        let risk_matrix = risks?;

        if risk_matrix.is_empty() || grid.is_empty() {
            return Err(BoostlssError::DataError(
                "Risk matrix or grid is empty".into(),
            ));
        }

        let mut mean_risks = vec![0.0; grid.len()];
        for fold_risks in &risk_matrix {
            for (m_idx, &r) in fold_risks.iter().enumerate() {
                mean_risks[m_idx] += r;
            }
        }
        let n_folds = risk_matrix.len() as f64;
        for r in &mut mean_risks {
            *r /= n_folds;
        }

        let mut min_idx = 0;
        let mut min_val = f64::INFINITY;
        for (i, &val) in mean_risks.iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        let optimal_mstop = grid[min_idx].clone();

        Ok(CvRiskResult {
            risk_matrix,
            mean_risk: mean_risks,
            optimal_mstop,
            mstop_grid: grid,
        })
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

    #[test]
    fn test_make_grid() {
        let grid = make_grid(2, 10, 3);
        // length_out = 3, min = 1, max = 10. log-spaced rounded: 1, 3, 10.
        // grid size = 3 * 3 = 9
        assert_eq!(grid.len(), 9);
        assert!(matches!(&grid[0], Mstop::PerParam(v) if v == &vec![1, 1]));
    }

    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use ndarray::{array, Array2};

    #[test]
    fn test_cv_risk_run() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .on("sigma", |p| p.add(Linear::new("x").intercept(true)))
            .unwrap()
            .mstop(Mstop::Scalar(10));

        let cv = CvRisk::new(model, Resampling::KFold { k: 2 })
            .mstop_max(3)
            .grid_resolution(2);

        let result = cv.run(&data).unwrap();

        assert_eq!(result.risk_matrix.len(), 2);
        assert_eq!(result.risk_matrix[0].len(), result.mstop_grid.len());
    }
}
