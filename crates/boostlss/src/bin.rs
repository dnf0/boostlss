use faer::linalg::solvers::Llt;
use faer::prelude::*;
use faer::Mat;

fn main() {
    let p = 2;
    let mut xtx = Mat::<f64>::zeros(p, p);
    xtx[(0, 0)] = 2.0;
    xtx[(1, 1)] = 2.0;

    let cholesky = xtx.cholesky(faer::Side::Lower).unwrap();
    let l = cholesky.compute_l();

    let mut k = Mat::<f64>::zeros(p, p);
    k[(0, 0)] = 1.0;
    k[(1, 1)] = 1.0;

    // L^{-1} K L^{-T}
    // we can use faer::linalg::solvers::Solve
    // l.solve_lower_triangular(...)
}
