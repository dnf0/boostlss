use crate::data::Dataset;
use crate::error::BoostlssError;
use crate::family::Family;
use crate::model::{BoostLss, Fitted};

pub fn fit_noncyclical<F: Family + Clone>(
    _model: BoostLss<F>,
    _data: &Dataset,
) -> Result<Fitted<F>, BoostlssError> {
    Err(BoostlssError::NotConverged(
        "NonCyclic fit unimplemented".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::engine::Mstop;
    use crate::family::GaussianLss;
    use crate::learner::Linear;
    use ndarray::array;

    #[test]
    fn test_fit_noncyclical_unimplemented() {
        let x = array![[1.0], [2.0]];
        let y = array![2.0, 4.0];
        let data = Dataset::new(x, y, None).unwrap();

        let model = BoostLss::new(GaussianLss::new())
            .on("mu", |p| p.add(Linear::new("x")))
            .unwrap()
            .algorithm(crate::engine::Algorithm::NonCyclic)
            .mstop(Mstop::Scalar(1));

        let res = fit_noncyclical(model, &data);
        assert!(res.is_err());
    }
}
