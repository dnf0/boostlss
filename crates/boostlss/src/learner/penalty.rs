use ndarray::Array2;

/// Create a difference matrix `D` of order `d` for `n` columns.
/// Penalty matrix K = D^T D.
pub fn difference_matrix(n: usize, d: usize) -> Array2<f64> {
    if d == 0 {
        return Array2::eye(n);
    }
    let prev = difference_matrix(n, d - 1);
    let mut out = Array2::zeros((prev.nrows() - 1, n));
    for i in 0..out.nrows() {
        for j in 0..n {
            out[[i, j]] = prev[[i + 1, j]] - prev[[i, j]];
        }
    }
    out
}

/// Compute K = D^T D
pub fn penalty_matrix(n: usize, d: usize) -> Array2<f64> {
    let diff = difference_matrix(n, d);
    let p = diff.ncols();
    let mut k = Array2::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            for r in 0..diff.nrows() {
                k[[i, j]] += diff[[r, i]] * diff[[r, j]];
            }
        }
    }
    k
}

/// Helper to map df to lambda. For v1, we use a simple heuristic or fallback.
pub fn df_to_lambda(_xtx: &Array2<f64>, _k: &Array2<f64>, _target_df: f64) -> f64 {
    // Exact Demmler-Reinsch eigenvalue solve requires faer symmetric eigensolver.
    // In v1, we provide a placeholder constant if requested df is fixed.
    1.0
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
}
