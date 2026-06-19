use faer::Mat;

#[test]
fn test_faer_basic() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 1.0;
    println!("{:?}", a);
}
