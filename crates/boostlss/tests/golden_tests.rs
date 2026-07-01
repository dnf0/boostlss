use approx::assert_relative_eq;
use boostlss::data::Dataset;
use boostlss::family::{Family, GaussianLss};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct InputData {
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct Offsets {
    mu: f64,
    sigma: f64,
}

#[test]
fn test_golden_offsets_match_r() {
    // 1. Read synthetic input data
    // (If the file doesn't exist, we skip the test to avoid failing in environments without R)
    let input_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/input_data.json"
    );
    let offsets_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/offsets.json");

    if !std::path::Path::new(input_path).exists() || !std::path::Path::new(offsets_path).exists() {
        println!("Fixtures not found, skipping golden test.");
        return;
    }

    let input_json = fs::read_to_string(input_path).unwrap();
    let data: InputData = serde_json::from_str(&input_json).unwrap();

    let offsets_json = fs::read_to_string(offsets_path).unwrap();
    let expected_offsets: Offsets = serde_json::from_str(&offsets_json).unwrap();

    // 2. Initialize our Rust family
    let gaussian = GaussianLss::new();

    // NOTE: The spec originally requested:
    // let weights = vec![1.0; data.y.len()];
    // let rust_offsets = gaussian.init_offsets(&data.y, Some(&weights));
    // However, the `init_offsets` signature was changed to require a `&Dataset` instead of `&Vec<f64>`.
    // We construct a dummy design matrix and a `Dataset` here to accommodate the new API.
    let n = data.y.len();
    let design = Array2::<f64>::zeros((n, 1));
    let response = Array1::from_vec(data.y);
    let weights = Array1::from_elem(n, 1.0);
    let dataset = Dataset::new(design, response, Some(weights), None).unwrap();

    // 3. Compute initial offsets in Rust
    let rust_offsets = gaussian.init_offsets(&dataset).unwrap();

    // 4. Assert mathematical equality
    // gamboostlss returns scalar offsets for intercept-only start
    assert_relative_eq!(rust_offsets[0], expected_offsets.mu, epsilon = 1e-6);
    assert_relative_eq!(rust_offsets[1], expected_offsets.sigma, epsilon = 1e-6);
}
