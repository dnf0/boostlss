use crate::util::weighted_mean;
use ndarray::Array1;

pub fn stabilize(u: &mut Array1<f64>, method: super::Stabilization, w: Option<&Array1<f64>>) {
    if u.is_empty() {
        return;
    }

    match method {
        super::Stabilization::None => {}
        super::Stabilization::Mad => {
            // Simplified MAD without weights for now, just to stub
            // Needs robust weighted median in later PR.
            // TODO: Using `mean` here as a temporary substitute for the `median`.
            let mean = weighted_mean(u, w);
            let mut diffs: Vec<f64> = u.iter().map(|&x| (x - mean).abs()).collect();
            diffs.sort_by(|a, b| a.total_cmp(b));
            let mad = diffs[diffs.len() / 2].max(1e-4);
            *u /= mad;
        }
        super::Stabilization::L2 => {
            let sq = u.mapv(|x| x * x);
            let rms = weighted_mean(&sq, w).sqrt().clamp(1e-4, 1e4);
            *u /= rms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empty_array() {
        let mut u = Array1::<f64>::zeros(0);
        stabilize(&mut u, crate::engine::Stabilization::None, None);
        assert!(u.is_empty());

        stabilize(&mut u, crate::engine::Stabilization::Mad, None);
        assert!(u.is_empty());

        stabilize(&mut u, crate::engine::Stabilization::L2, None);
        assert!(u.is_empty());
    }

    #[test]
    fn test_stabilize_none() {
        let mut u = array![1.0, 2.0, 3.0];
        let original = u.clone();
        stabilize(&mut u, crate::engine::Stabilization::None, None);
        assert_eq!(u, original);
    }

    #[test]
    fn test_stabilize_mad() {
        let mut u = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = u.clone();
        stabilize(&mut u, crate::engine::Stabilization::Mad, None);
        assert_eq!(u, original);

        let mut u2 = array![1.0, 1.0, 1.0, 5.0, 5.0];
        stabilize(&mut u2, crate::engine::Stabilization::Mad, None);
        assert_eq!(
            u2,
            array![1.0 / 1.6, 1.0 / 1.6, 1.0 / 1.6, 5.0 / 1.6, 5.0 / 1.6]
        );
    }

    #[test]
    fn test_stabilize_l2() {
        let mut u = array![3.0, 4.0];
        let rms = 12.5_f64.sqrt();
        stabilize(&mut u, crate::engine::Stabilization::L2, None);
        assert_eq!(u, array![3.0 / rms, 4.0 / rms]);
    }
}
