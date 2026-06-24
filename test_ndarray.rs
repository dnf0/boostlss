fn main() {
    let array = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
    let view = array.view();
    let owned: ndarray::Array2<f64> = view.to_owned();
}
