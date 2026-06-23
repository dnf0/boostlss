//! Distributional families for boosting.

use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::param::ParamSpec;
use ndarray::Array1;

pub mod beta;
pub mod binomial;
pub mod gamma;
pub mod gaussian;
pub mod nbinomial;
pub mod student_t;
pub mod weibull;

pub use beta::BetaLss;
pub use binomial::BinomialLss;
pub use gamma::GammaLss;
pub use gaussian::GaussianLss;
pub use nbinomial::NBinomialLss;
pub use student_t::StudentTLss;
pub use weibull::WeibullLss;

pub trait Family: std::fmt::Debug {
    /// Information about the parameters of this family, in fixed order.
    fn params(&self) -> &[ParamSpec];

    /// Evaluate the negative log-likelihood of the observations.
    fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError>;

    /// Compute the negative gradient w.r.t the k-th parameter's additive predictor.
    /// Default implementation uses finite differences on `nll`.
    fn ngradient(
        &self,
        data: &Dataset,
        eta: &[Array1<f64>],
        k: usize,
    ) -> Result<Array1<f64>, BoostlssError> {
        let eps = 1e-5;
        let mut eta_plus = eta.to_vec();
        let mut eta_minus = eta.to_vec();

        let mut grad = Array1::zeros(data.n_obs());
        for i in 0..data.n_obs() {
            eta_plus[k][i] += eps;
            eta_minus[k][i] -= eps;

            let l_plus = self.nll(data, &eta_plus)?;
            let l_minus = self.nll(data, &eta_minus)?;

            // central difference: (f(x+h) - f(x-h)) / 2h
            // We want the negative gradient: - d(NLL)/d(eta)
            grad[i] = -(l_plus - l_minus) / (2.0 * eps);

            eta_plus[k][i] -= eps; // reset
            eta_minus[k][i] += eps; // reset
        }
        Ok(grad)
    }

    /// Default initialization offsets for the parameters (on the additive predictor scale).
    fn init_offsets(&self, data: &Dataset) -> Result<Vec<f64>, BoostlssError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::IdentityLink;
    use ndarray::{array, Array2};

    #[derive(Debug)]
    struct DummyFamily {
        params: Vec<ParamSpec>,
    }

    impl DummyFamily {
        fn new() -> Self {
            Self {
                params: vec![ParamSpec::new("mu", IdentityLink)],
            }
        }
    }

    impl Family for DummyFamily {
        fn params(&self) -> &[ParamSpec] {
            &self.params
        }

        // A dummy NLL: sum(0.5 * (y - eta[0])^2)
        fn nll(&self, data: &Dataset, eta: &[Array1<f64>]) -> Result<f64, BoostlssError> {
            let diff = data.response() - &eta[0];
            Ok(0.5 * diff.mapv(|x| x * x).sum())
        }

        fn init_offsets(&self, _data: &Dataset) -> Result<Vec<f64>, BoostlssError> {
            Ok(vec![0.0])
        }
    }

    #[test]
    fn default_ngradient_is_accurate() {
        let fam = DummyFamily::new();
        let ds = Dataset::new(Array2::<f64>::zeros((2, 1)), array![1.0, 3.0], None).unwrap();
        let eta = vec![array![0.0, 0.0]];

        let grad = fam.ngradient(&ds, &eta, 0).unwrap();

        // Exact negative gradient of 0.5(y-eta)^2 wrt eta is (y - eta)
        assert!(approx::relative_eq!(grad[0], 1.0, epsilon = 1e-4));
        assert!(approx::relative_eq!(grad[1], 3.0, epsilon = 1e-4));
    }
}
