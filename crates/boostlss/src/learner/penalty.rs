use faer::{Mat, Side};
use ndarray::{s, Array2};

/// Create a difference matrix `D` of order `d` for `n` columns.
/// Penalty matrix K = D^T D.
pub fn difference_matrix(n: usize, d: usize) -> Array2<f64> {
    if d == 0 {
        return Array2::eye(n);
    }
    let prev = difference_matrix(n, d - 1);
    let nrows = prev.nrows();
    &prev.slice(s![1.., ..]) - &prev.slice(s![..nrows - 1, ..])
}

/// Compute K = D^T D
pub fn penalty_matrix(n: usize, d: usize) -> Array2<f64> {
    let diff = difference_matrix(n, d);
    diff.t().dot(&diff)
}

/// Helper to map df to lambda using Demmler-Reinsch orthogonalization.
pub fn df_to_lambda(xtx: &Array2<f64>, k: &Array2<f64>, target_df: f64) -> f64 {
    let p = xtx.ncols();

    // Add small ridge to xtx for stability
    let a = Mat::from_fn(p, p, |i, j| xtx[[i, j]] + if i == j { 1e-8 } else { 0.0 });

    let eig_a = match a.self_adjoint_eigen(Side::Lower) {
        Ok(eig) => eig,
        Err(_) => return 1.0, // Fallback
    };

    let u_faer = eig_a.U();
    let s_faer = eig_a.S();

    let u = Array2::from_shape_fn((p, p), |(i, j)| u_faer[(i, j)]);
    let s_inv_half = ndarray::Array1::from_shape_fn(p, |i| {
        let val = s_faer[i];
        if val > 1e-12 {
            1.0 / val.sqrt()
        } else {
            0.0
        }
    });

    // a_inv_half = U * diag(S^{-1/2}) * U^T
    let mut u_scaled = u.clone();
    for j in 0..p {
        let scale = s_inv_half[j];
        for i in 0..p {
            u_scaled[[i, j]] *= scale;
        }
    }
    let a_inv_half = u_scaled.dot(&u.t());

    // M = A^{-1/2} K A^{-1/2}
    let m_nd = a_inv_half.dot(k).dot(&a_inv_half);

    let m_faer = Mat::from_fn(p, p, |i, j| m_nd[[i, j]]);
    let eig_m = match m_faer.self_adjoint_eigen(Side::Lower) {
        Ok(eig) => eig,
        Err(_) => return 1.0,
    };

    let d = ndarray::Array1::from_shape_fn(p, |i| eig_m.S()[i]);

    // Max df is p (when lambda = 0). If target is higher, lambda = 0.
    if target_df >= p as f64 - 1e-4 {
        return 0.0;
    }

    let mut low = 0.0;
    let mut high = 1.0;

    // Expand high if necessary
    for _ in 0..50 {
        let df_high: f64 = d.iter().map(|&di| 1.0 / (1.0 + high * di)).sum();
        if df_high <= target_df {
            break;
        }
        high *= 10.0;
    }

    // Bisection
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let df: f64 = d.iter().map(|&di| 1.0 / (1.0 + mid * di)).sum();

        if df > target_df {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_difference_matrix_d1() {
        let diff = difference_matrix(3, 1);
        let expected = array![[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]];
        assert_eq!(diff, expected);
    }

    #[test]
    fn test_penalty_matrix() {
        let pen = penalty_matrix(3, 1);
        let diff = difference_matrix(3, 1);
        assert_eq!(pen, diff.t().dot(&diff));
    }

    #[test]
    fn test_df_to_lambda() {
        let xtx = Array2::eye(3);
        let k = Array2::eye(3);
        // target df = 1.5 -> each lambda should be the same
        // 3 / (1 + lambda) = 1.5 => 1 + lambda = 2 => lambda = 1.0
        let lambda = df_to_lambda(&xtx, &k, 1.5);
        assert!((lambda - 1.0).abs() < 1e-4);
    }
}
