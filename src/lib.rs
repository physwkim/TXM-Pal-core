use rayon::prelude::*;
use ndarray::{ Array, s };
mod fit;
use fit::{ quadratic_fit_center, gaussian_fit_center };

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use pyo3::wrap_pyfunction;
use std::sync::Mutex;
use std::f64::NAN;

#[pyfunction]
fn quadfit(py: Python, energy: &PyArray1<f64>, image: &PyArray3<f64>, points: &PyInt, mask: &PyArray2<u8>) -> PyResult<Py<PyArray2<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let stack = unsafe { image.as_array() };
    let mask = unsafe { mask.as_array() };
    let num_points = points.extract::<usize>()?;

    let shape = stack.shape();
    let mut result = Array::zeros((shape[1], shape[2]));

    for i in 0..shape[1] {
        for j in 0..shape[2] {
            let slice = stack.slice(s![.., i, j]);

            let (max_idx, _min_idx) = slice.iter()
                                          .cloned()
                                          .enumerate()
                                          .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                          .unwrap();

            let half_points = num_points / 2;

            let mut start_idx = 0;
            if max_idx > half_points {
                start_idx = max_idx - half_points;
            }

            let mut end_idx = max_idx + half_points + 1;
            if end_idx > slice.len() - 1 {
                end_idx = slice.len() - 1;
            }

            // initial_guessing
            // guess c
            let c = slice[start_idx];

            // guess b
            let b = (slice[end_idx] - slice[start_idx]) / (nrj[end_idx] - nrj[start_idx]);

            // guess a (should be negative)
            let mut a = -b/(2.0 * nrj[max_idx]);
            if a > 0.0 {
                a = -a;
            }

            let initial_guess = vec![a, b, c];
            let xdata = nrj.slice(s![start_idx..end_idx]).to_vec();
            let ydata = slice.slice(s![start_idx..end_idx]).to_vec();
            if mask[[i, j]] > 0 {
                result[[i, j]] = quadratic_fit_center(xdata, ydata, initial_guess);
            } else {
                result[[i, j]] = NAN;
            }
        }
    }
    let py_result = PyArray2::from_array(py, &result);
    Ok(py_result.to_owned())
}

#[pyfunction]
fn quadfit_mc(py: Python, energy: &PyArray1<f64>, image: &PyArray3<f64>, points: &PyInt, mask: &PyArray2<u8>) -> PyResult<Py<PyArray2<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let stack = unsafe { image.as_array() };
    let num_points = points.extract::<usize>()?;
    let mask = unsafe { mask.as_array() };

    let shape = stack.shape();
    let mut result = Mutex::new(Array::zeros((shape[1], shape[2])));

    (0..shape[1]).into_par_iter().for_each(|i| {
        for j in 0..shape[2] {
            let slice = stack.slice(s![.., i, j]);

            let (max_idx, _min_idx) = slice.iter()
                                          .cloned()
                                          .enumerate()
                                          .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                          .unwrap();

            let half_points = num_points / 2;

            let mut start_idx = 0;
            if max_idx > half_points {
                start_idx = max_idx - half_points;
            }

            let mut end_idx = max_idx + half_points + 1;
            if end_idx > slice.len() - 1 {
                end_idx = slice.len() - 1;
            }

            // initial_guessing (ax^2 + bx + c)
            // guess c
            let c = slice[start_idx];

            // guess b
            let b = (slice[end_idx] - slice[start_idx]) / (nrj[end_idx] - nrj[start_idx]);

            // guess a (should be negative)
            let mut a = -b/(2.0 * nrj[max_idx]);
            if a > 0.0 {
                a = -a;
            }

            let initial_guess = vec![a, b, c];
            let xdata = nrj.slice(s![start_idx..end_idx]).to_vec();
            let ydata = slice.slice(s![start_idx..end_idx]).to_vec();
            let mut guarded_result = result.lock().unwrap();
            if mask[[i, j]] > 0 {
                guarded_result[[i, j]] = quadratic_fit_center(xdata, ydata, initial_guess);
            } else {
                guarded_result[[i, j]] = NAN;
            }
        }
    });
    let py_result = PyArray2::from_array(py, &*result.lock().unwrap());
    Ok(py_result.to_owned())
}

#[pyfunction]
fn gaussianfit(py: Python, energy: &PyArray1<f64>, image: &PyArray3<f64>, points: &PyInt, mask: &PyArray2<u8>) -> PyResult<Py<PyArray2<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let (nrjmin, nrjmax) = nrj.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
        (min.min(val), max.max(val))
    });
    let stack = unsafe { image.as_array() };
    let mask = unsafe { mask.as_array() };
    let num_points = points.extract::<usize>()?;

    let shape = stack.shape();
    let mut result = Array::zeros((shape[1], shape[2]));

    for i in 0..shape[1] {
        for j in 0..shape[2] {
            let slice = stack.slice(s![.., i, j]);

            let (max_idx, _) = slice.iter()
                                    .cloned()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .unwrap();

            let (min_idx, _) = slice.iter()
                                    .cloned()
                                    .enumerate()
                                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .unwrap();

            // initial_guessing (a*exp(-(x-b)^2/(2*c^2))
            let maxy = slice[max_idx];
            let miny = slice[min_idx];
            let maxx = nrjmax;
            let minx = nrjmin;
            let cen = nrj[max_idx];
            let height = (maxy - miny) * 3.0;
            let sig = (maxx - minx) / 6.0;
            let amp = height * sig;

            let half_points = num_points / 2;

            let mut start_idx = 0;
            if max_idx > half_points {
                start_idx = max_idx - half_points;
            }

            let mut end_idx = max_idx + half_points + 1;
            if end_idx > slice.len() - 1 {
                end_idx = slice.len() - 1;
            }

            let initial_guess = vec![amp, cen, sig];
            let xdata = nrj.slice(s![start_idx..end_idx]).to_vec();
            let ydata = slice.slice(s![start_idx..end_idx]).to_vec();

            if mask[[i, j]] > 0 {
                result[[i, j]] = gaussian_fit_center(xdata, ydata, initial_guess);
            } else {
                result[[i, j]] = NAN;
            }


        }
    }
    let py_result = PyArray2::from_array(py, &result);
    Ok(py_result.to_owned())
}

#[pymodule]
fn lmfitrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quadfit, m)?)?;
    m.add_function(wrap_pyfunction!(gaussianfit, m)?)?;
    m.add_function(wrap_pyfunction!(quadfit_mc, m)?)?;

    Ok(())
}
