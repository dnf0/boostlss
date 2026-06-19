//! Small numeric helpers shared across families.

use ndarray::Array1;

/// Weighted mean of `y`. With `w = None`, the ordinary mean.
pub fn weighted_mean(y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
    match w {
        None => y.sum() / y.len() as f64,
        Some(w) => {
            let sw: f64 = w.sum();
            y.iter().zip(w.iter()).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        }
    }
}

/// Weighted sample standard deviation (denominator = effective n - 1).
/// With `w = None` this is the ordinary sample standard deviation.
pub fn weighted_sd(y: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
    let m = weighted_mean(y, w);
    match w {
        None => {
            let n = y.len() as f64;
            let ss: f64 = y.iter().map(|yi| (yi - m).powi(2)).sum();
            (ss / (n - 1.0)).sqrt()
        }
        Some(w) => {
            let sw: f64 = w.sum();
            let ss: f64 = y
                .iter()
                .zip(w.iter())
                .map(|(yi, wi)| wi * (yi - m).powi(2))
                .sum();
            (ss / (sw - 1.0)).sqrt()
        }
    }
}

/// Minimize a unimodal `f` on `[lo, hi]` by golden-section search.
/// Used for intercept-only MLE offsets that have no closed form.
pub fn minimize_1d<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_8; // 1/golden ratio
    const ITERS: usize = 100;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - (b - a) * INV_PHI;
    let mut d = a + (b - a) * INV_PHI;
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..ITERS {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INV_PHI;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INV_PHI;
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn weighted_mean_unweighted_is_arithmetic_mean() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(weighted_mean(&y, None), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn weighted_mean_respects_weights() {
        let y = array![1.0, 3.0];
        let w = array![3.0, 1.0];
        assert_relative_eq!(weighted_mean(&y, Some(&w)), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn weighted_sd_unweighted_is_sample_sd() {
        let y = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // sample sd (n-1) of this classic set is exactly sqrt(32/7).
        assert_relative_eq!(
            weighted_sd(&y, None),
            (32.0_f64 / 7.0).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn minimize_1d_finds_parabola_vertex() {
        let x = minimize_1d(|x| (x - 3.0).powi(2) + 1.0, -10.0, 10.0);
        assert_relative_eq!(x, 3.0, epsilon = 1e-6);
    }
}
