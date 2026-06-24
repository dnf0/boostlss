use crate::error::BoostlssError;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineData {
    pub min_val: f64,
    pub max_val: f64,
    pub t: Vec<f64>,
}

pub fn build_bspline_design(
    x: &Array1<f64>,
    knots: usize,
    degree: usize,
    spline_data: &mut Option<SplineData>,
) -> Result<Array2<f64>, BoostlssError> {
    if spline_data.is_none() {
        let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let num_knots = knots + 2 * degree + 2;
        let mut t = vec![0.0; num_knots];
        let step = (max_val - min_val) / (knots as f64 - 1.0 + 1e-9);

        for (i, t_val) in t.iter_mut().enumerate() {
            *t_val = min_val + (i as f64 - degree as f64) * step;
        }

        *spline_data = Some(SplineData {
            min_val,
            max_val,
            t,
        });
    }

    let data = spline_data.as_ref().unwrap();
    let num_knots = data.t.len();
    let n = x.len();
    let p = knots + degree + 1;
    let mut b = Array2::zeros((n, p));

    let mut n_basis = vec![0.0; num_knots - 1];
    let mut next_n = vec![0.0; num_knots - 1];

    for i in 0..n {
        let xi = x[i];
        if xi < data.min_val || xi > data.max_val {
            return Err(BoostlssError::OutOfRange(format!(
                "Value {} out of training range",
                xi
            )));
        }

        n_basis.fill(0.0);
        for j in 0..num_knots - 1 {
            if data.t[j] <= xi && xi < data.t[j + 1] {
                n_basis[j] = 1.0;
            }
        }
        if xi == data.t[num_knots - 1] {
            n_basis[num_knots - 2] = 1.0;
        }

        for d in 1..=degree {
            next_n.fill(0.0);
            for j in 0..num_knots - 1 - d {
                let left_den = data.t[j + d] - data.t[j];
                let left = if left_den > 0.0 {
                    (xi - data.t[j]) / left_den * n_basis[j]
                } else {
                    0.0
                };

                let right_den = data.t[j + d + 1] - data.t[j + 1];
                let right = if right_den > 0.0 {
                    (data.t[j + d + 1] - xi) / right_den * n_basis[j + 1]
                } else {
                    0.0
                };
                next_n[j] = left + right;
            }
            n_basis.copy_from_slice(&next_n);
        }

        for j in 0..p {
            b[[i, j]] = n_basis[j];
        }
    }

    Ok(b)
}
