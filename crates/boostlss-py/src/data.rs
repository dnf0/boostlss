use boostlss::data::SparseMatrix;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub fn extract_sparse<'py>(
    _py: Python<'py>,
    sparse_obj: &Bound<'py, PyAny>,
) -> PyResult<SparseMatrix> {
    let shape_tuple = sparse_obj.getattr("shape")?.extract::<(usize, usize)>()?;

    let data_arr: PyReadonlyArray1<f64> = sparse_obj.getattr("data")?.extract()?;
    let indices_arr: PyReadonlyArray1<i32> = sparse_obj.getattr("indices")?.extract()?;
    let indptr_arr: PyReadonlyArray1<i32> = sparse_obj.getattr("indptr")?.extract()?;

    let data_view = data_arr.as_array();
    let data =
        ndarray::Array1::from_shape_vec((data_view.len(),), data_view.to_owned().into_raw_vec())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let indices_view = indices_arr.as_array();
    let indices = ndarray::Array1::from_shape_vec(
        (indices_view.len(),),
        indices_view.mapv(|x| x as usize).into_raw_vec(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let indptr_view = indptr_arr.as_array();
    let indptr = ndarray::Array1::from_shape_vec(
        (indptr_view.len(),),
        indptr_view.mapv(|x| x as usize).into_raw_vec(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    SparseMatrix::new(data, indices, indptr, shape_tuple)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

pub fn extract_dataset<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    optional_y: Option<ndarray::Array1<f64>>,
) -> PyResult<boostlss::data::Dataset> {
    let is_sparse = x.hasattr("format")?;

    if is_sparse {
        let format: String = x.getattr("format")?.extract()?;
        let sparse_mat = extract_sparse(py, x)?;
        let n_obs = sparse_mat.shape.0;
        let y = optional_y.unwrap_or_else(|| ndarray::Array1::zeros(n_obs));

        match format.as_str() {
            "csr" => boostlss::data::Dataset::new_csr(sparse_mat, y, None, None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            "csc" => boostlss::data::Dataset::new_csc(sparse_mat, y, None, None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Only CSR and CSC formats are supported",
            )),
        }
    } else {
        let x_dense: numpy::PyReadonlyArray2<f64> = x.extract()?;
        let x_view = x_dense.as_array();
        let x_mat = ndarray::Array2::from_shape_vec(
            (x_view.nrows(), x_view.ncols()),
            x_view.to_owned().into_raw_vec(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let y = optional_y.unwrap_or_else(|| ndarray::Array1::zeros(x_mat.nrows()));
        boostlss::data::Dataset::new(x_mat, y, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}
