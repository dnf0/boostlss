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
            .unwrap();

    let indices_view = indices_arr.as_array();
    let indices = ndarray::Array1::from_shape_vec(
        (indices_view.len(),),
        indices_view.mapv(|x| x as usize).into_raw_vec(),
    )
    .unwrap();

    let indptr_view = indptr_arr.as_array();
    let indptr = ndarray::Array1::from_shape_vec(
        (indptr_view.len(),),
        indptr_view.mapv(|x| x as usize).into_raw_vec(),
    )
    .unwrap();

    SparseMatrix::new(data, indices, indptr, shape_tuple)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
