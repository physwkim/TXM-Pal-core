use ndarray::prelude::*;
use rayon::prelude::*;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use pyo3::wrap_pyfunction;

use std::f64::NAN;

// use whittaker_smoother::whittaker_smoother;
use savgol_rs::{savgol_filter, SavGolInput};

mod filter;
use filter::{boxcar, medfilt, multi_3point_average};

mod fit;
use fit::{gaussian_fit_center, quadratic_fit, quadratic_fit_center};

mod phase_cross_correlation;
use phase_cross_correlation::phase_cross_correlation;

use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use std::sync::{Arc, Mutex};

#[pyfunction]
fn quadfit_mc(
    py: Python,
    energy: &PyArray1<f64>,
    image: &PyArray3<f64>,
    points: &PyInt,
    mask: &PyArray2<u8>,
    start_e: &PyFloat,
    stop_e: &PyFloat,
    smooth: &PyBool,
    algo: &PyString,
    smooth_width: &PyInt,
    smooth_order: &PyInt,
) -> PyResult<Py<PyArray2<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let start_e = start_e.value();
    let stop_e = stop_e.value();

    let smooth_width: usize = smooth_width.extract::<usize>()?;
    let smooth_order: usize = smooth_order.extract::<usize>()?;

    let start_idx = nrj.iter().position(|&x| x >= start_e).unwrap();
    let stop_idx = nrj.iter().position(|&x| x >= stop_e).unwrap();

    let stack = unsafe { image.as_array() };
    let num_points = points.extract::<usize>()?;
    let mask = unsafe { mask.as_array() };
    let smooth = smooth.extract::<bool>()?;
    let algorithm: &str = algo.extract::<&PyString>()?.to_str()?;

    let shape = stack.shape();

    let final_result = py.allow_threads(|| {
        let result: Vec<_> = (0..shape[1])
            .into_par_iter()
            .map(|i| {
                let mut local_result = Array::zeros((shape[1], shape[2]));
                for j in 0..shape[2] {
                    let mut slice = stack.slice(s![.., i, j]).to_vec();

                    // smoothing
                    if smooth {
                        if algorithm == "savgol" {
                            let input = SavGolInput {
                                data: &slice,
                                window_length: smooth_width,
                                poly_order: smooth_order,
                                derivative: 0,
                            };
                            slice = savgol_filter(&input).unwrap();
                        } else if algorithm == "median" {
                            slice = medfilt(slice, smooth_width, "zeropadding");
                        } else if algorithm == "3point" {
                            slice = multi_3point_average(&slice, smooth_width);
                        } else if algorithm == "boxcar" {
                            slice = boxcar(&slice, smooth_width);
                        }
                    }

                    let sub_slice = &slice[start_idx..stop_idx];
                    let (relative_max_idx, _max_value) = sub_slice
                        .iter()
                        .cloned()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    // to ndarray
                    let slice = Array::from(slice);
                    let half_points = num_points / 2;

                    let max_idx = relative_max_idx + start_idx;

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
                    let mut a = -b / (2.0 * nrj[max_idx]);
                    if a > 0.0 {
                        a = -a;
                    }

                    let initial_guess = vec![a, b, c];
                    let xdata = nrj.slice(s![start_idx..end_idx]).to_vec();
                    let ydata = slice.slice(s![start_idx..end_idx]).to_vec();

                    if mask[[i, j]] > 0 {
                        local_result[[i, j]] = quadratic_fit_center(xdata, ydata, initial_guess);
                    } else {
                        local_result[[i, j]] = NAN;
                    }
                }
                local_result
            })
            .collect();

        result
            .iter()
            .fold(Array::zeros((shape[1], shape[2])), |acc, arr| acc + arr)
    });

    let py_result = PyArray2::from_array(py, &final_result);
    Ok(py_result.to_owned())
}

#[pyfunction]
fn quadfit_single(
    _py: Python,
    xdata: &PyArray1<f64>,
    ydata: &PyArray1<f64>,
    initial_guess: &PyArray1<f64>,
) -> PyResult<f64> {
    let xdata = unsafe { xdata.as_array() };
    let ydata = unsafe { ydata.as_array() };
    let initial_guess = unsafe { initial_guess.as_array() };

    let result = quadratic_fit(xdata.to_vec(), ydata.to_vec(), initial_guess.to_vec());
    println!("{:?}", result);
    Ok(result[1])
}

#[pyfunction]
fn gaussianfit_mc(
    py: Python,
    energy: &PyArray1<f64>,
    image: &PyArray3<f64>,
    points: &PyInt,
    mask: &PyArray2<u8>,
    start_e: &PyFloat,
    stop_e: &PyFloat,
    smooth: &PyBool,
    algo: &PyString,
    smooth_width: &PyInt,
    smooth_order: &PyInt,
) -> PyResult<Py<PyArray2<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let start_e = start_e.value();
    let stop_e = stop_e.value();
    let smooth_width: usize = smooth_width.extract::<usize>()?;
    let smooth_order: usize = smooth_order.extract::<usize>()?;

    let start_idx = nrj.iter().position(|&x| x >= start_e).unwrap();
    let stop_idx = nrj.iter().position(|&x| x >= stop_e).unwrap();

    let (nrjmin, nrjmax) = nrj.to_vec()[start_idx..stop_idx]
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (min.min(val), max.max(val))
        });
    let stack = unsafe { image.as_array() };
    let mask = unsafe { mask.as_array() };
    let num_points = points.extract::<usize>()?;
    let smooth = smooth.extract::<bool>()?;
    let algorithm: &str = algo.extract::<&PyString>()?.to_str()?;

    let shape = stack.shape();

    let final_result = py.allow_threads(|| {
        let result: Vec<_> = (0..shape[1])
            .into_par_iter()
            .map(|i| {
                let mut local_result = Array::zeros((shape[1], shape[2]));
                for j in 0..shape[2] {
                    let mut slice = stack.slice(s![.., i, j]).to_vec();

                    // smoothing
                    if smooth {
                        if algorithm == "savgol" {
                            let input = SavGolInput {
                                data: &slice,
                                window_length: smooth_width,
                                poly_order: smooth_order,
                                derivative: 0,
                            };
                            slice = savgol_filter(&input).unwrap();
                        } else if algorithm == "medfilt" {
                            slice = medfilt(slice, smooth_width, "zeropadding");
                        } else if algorithm == "3point" {
                            slice = multi_3point_average(&slice, smooth_width);
                        } else if algorithm == "boxcar" {
                            slice = boxcar(&slice, smooth_width);
                        }
                    }
                    let sub_slice = &slice[start_idx..stop_idx];
                    let (relative_max_idx, _) = &sub_slice
                        .iter()
                        .cloned()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    let (relative_min_idx, _) = &sub_slice
                        .iter()
                        .cloned()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    let max_idx = relative_max_idx + start_idx;
                    let min_idx = relative_min_idx + start_idx;

                    // initial_guessing (a*exp(-(x-b)^2/(2*c^2))
                    let maxy = slice[max_idx];
                    let miny = slice[min_idx];
                    let maxx = nrjmax;
                    let minx = nrjmin;
                    let cen = nrj[max_idx];
                    let height = (maxy - miny) * 3.0;
                    let sig = (maxx - minx) / 6.0;
                    let amp = height * sig;

                    // to ndarray
                    let slice = Array::from(slice);
                    let half_points = num_points / 2;

                    let mut start_idx = 0;
                    if max_idx > half_points {
                        start_idx = max_idx - half_points;
                    }

                    let mut end_idx = max_idx + half_points + 1;
                    if end_idx > slice.len() - 1 {
                        end_idx = slice.len() - 1;
                    }

                    let initial_guess = vec![amp, cen, sig, miny];
                    let xdata = nrj.slice(s![start_idx..end_idx]).to_vec();
                    let ydata = slice.slice(s![start_idx..end_idx]).to_vec();

                    if mask[[i, j]] > 0 {
                        local_result[[i, j]] = gaussian_fit_center(xdata, ydata, initial_guess);
                    } else {
                        local_result[[i, j]] = NAN;
                    }
                }
                local_result
            })
            .collect();

        result
            .iter()
            .fold(Array::zeros((shape[1], shape[2])), |acc, arr| acc + arr)
    });

    let py_result = PyArray2::from_array(py, &final_result);
    Ok(py_result.to_owned())
}

#[pyfunction]
fn phase_cross_correlation_rs(
    py: Python,
    reference_image: &PyArray2<f64>,
    moving_image: &PyArray2<f64>,
    upsample_factor: &PyInt,
) -> PyResult<Py<PyArray1<f64>>> {
    let reference_image = unsafe { reference_image.as_array() }.to_owned();
    let moving_image = unsafe { moving_image.as_array() }.to_owned();
    let upsample_factor = upsample_factor.extract::<usize>().unwrap();
    let shift_vals = phase_cross_correlation(&reference_image, &moving_image, upsample_factor);
    let shift_vals = array![shift_vals.0, shift_vals.1];
    let shift_vals_py = PyArray1::from_vec(py, shift_vals.to_vec());

    Ok(shift_vals_py.into())
}

#[pyfunction]
fn phase_cross_correlation_stack(
    py: Python,
    stack: &PyArray3<f64>,
    ref_index: &PyInt,
    upsample_factor: &PyInt,
) -> PyResult<Py<PyArray2<f64>>> {
    let stack = unsafe { stack.as_array() };
    let ref_index = ref_index.extract::<usize>()?;
    let upsample_factor = upsample_factor.extract::<usize>()?;

    let reference_image: Array2<f64> = stack.slice(s![ref_index, .., ..]).to_owned();

    let shifts: Vec<_> = py.allow_threads(|| {
        (0..stack.shape()[0])
            .into_par_iter()
            .map(|i| {
                let moving_image: Array2<f64> = stack.slice(s![i, .., ..]).to_owned();
                phase_cross_correlation(&reference_image, &moving_image, upsample_factor)
            })
            .collect()
    });

    let shifts_vec: Vec<Vec<f64>> = shifts.into_iter().map(|(x, y)| vec![x, y]).collect();
    let shifts_list = PyArray2::from_vec2(py, &shifts_vec)?;
    Ok(shifts_list.to_owned())
}

#[pyfunction]
fn renormalize_absorbance_stack(
    py: Python,
    energy: &PyArray1<f64>,
    image: &PyArray3<f64>,
    energy_offset_scale: &PyFloat,
) -> PyResult<Py<PyArray3<f64>>> {
    let nrj = unsafe { energy.as_array() };
    let offset_scale = energy_offset_scale.value();
    let stack = unsafe { image.as_array() };

    let shape = stack.shape();
    let stack_normalized = Arc::new(Mutex::new(Array3::<f64>::zeros((
        shape[0], shape[1], shape[2],
    ))));

    py.allow_threads(|| {
        (0..shape[1])
            .into_par_iter()
            .map(|i| {
                let mut local_results = Vec::new();
                for j in 0..shape[2] {
                    let off_energy = nrj.mapv(|x| x - i as f64 * offset_scale);
                    let interpolator = Interp1DBuilder::new(stack.slice(s![.., i, j]))
                        .x(off_energy.clone())
                        .strategy(Linear::new().extrapolate(true))
                        .build()
                        .unwrap();

                    let result: Vec<f64> = interpolator.interp_array(&nrj).unwrap().to_vec();
                    local_results.push((j, result));
                }
                (i, local_results)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(i, local_results)| {
                let mut stack_normalized_guard = stack_normalized.lock().unwrap();
                for (j, result) in local_results {
                    stack_normalized_guard
                        .slice_mut(s![.., i, j])
                        .assign(&Array::from(result));
                }
            });
    });

    let stack_normalized_array = stack_normalized.lock().unwrap();
    let py_result = PyArray3::from_array(py, &stack_normalized_array);
    Ok(py_result.to_owned())
}

fn gradient(data: &Vec<f64>) -> Vec<f64> {
    let n = data.len();
    let mut grad = Vec::with_capacity(n);

    if n < 2 {
        return vec![]; // 데이터가 너무 적으면 빈 벡터 반환
    }

    // 첫 번째 요소: forward difference (전진 차분)
    grad.push(data[1] - data[0]);

    // 중간 요소들: central difference (중앙 차분)
    for i in 1..n - 1 {
        let diff = (data[i + 1] - data[i - 1]) / 2.0;
        grad.push(diff);
    }

    // 마지막 요소: backward difference (후진 차분)
    grad.push(data[n - 1] - data[n - 2]);

    grad
}

#[pyfunction]
fn process_images(_py: Python, subdata: &PyArray3<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let subdata = unsafe { subdata.as_array() };
    let shape = subdata.shape();
    let mut centers = Vec::with_capacity(shape[0]);
    let mut yshifts = Vec::with_capacity(shape[0]);
    let mut center_shift = Vec::with_capacity(shape[0]);
    let mut y_shift = Vec::with_capacity(shape[0]);

    for i in 0..shape[0] {
        // x-axis sum for gaussian fitting
        let ydata = subdata.index_axis(Axis(0), i).sum_axis(Axis(0)).to_vec();
        let xdata: Vec<f64> = (0..ydata.len()).map(|x| x as f64).collect();

        let max_idx = ydata
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let min_idx = ydata
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        // initial_guessing (a*exp(-(x-b)^2/(2*c^2))
        let maxy = ydata[max_idx];
        let miny = ydata[min_idx];
        let cen = max_idx as f64;
        let height = (maxy - miny) * 3.0;
        let sig = (ydata.len() as f64) / 6.0;
        let amp = height * sig;

        let initial_guess = vec![amp, cen, sig, miny];

        // gaussian fitting
        let center = gaussian_fit_center(xdata, ydata, initial_guess);
        centers.push(center);

        // Y-axis line profile extraction and 3-point smoothing
        let center_idx = center.floor() as usize;

        // 인덱스 범위 조정
        let max_center_idx = subdata.shape()[2] - 2; // 배열의 최대 인덱스를 벗어나지 않도록 설정
        let center_idx = if center_idx < 1 {
            1
        } else if center_idx > max_center_idx {
            max_center_idx
        } else {
            center_idx
        };

        let start_idx = center_idx - 1;
        let end_idx = center_idx + 2; // 이미 max_center_idx로 제한했으므로 안전
        let profile = subdata.slice(s![i, .., start_idx..end_idx]).sum_axis(Axis(1)).to_vec();

        // Apply three point average
        let smoothed = multi_3point_average(&profile, 1);
        let profile_gradient: Vec<f64> = gradient(&smoothed).iter().map(|w| w.abs()).collect();
        let xdata: Vec<f64> = (0..profile_gradient.len()).map(|x| x as f64).collect();

        // Gaussian fitting for Y shift
        // Initial guessing
        let max_idx = profile_gradient
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let min_idx = profile_gradient
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let maxy = profile_gradient[max_idx];
        let miny = profile_gradient[min_idx];
        let cen = max_idx as f64;
        let height = (maxy - miny) * 3.0;
        let sig = (profile_gradient.len() as f64) / 6.0;
        let amp = height * sig;

        let initial_guess = vec![amp, cen, sig, miny];

        let yshift_fitted = gaussian_fit_center(xdata, profile_gradient, initial_guess);
        yshifts.push(yshift_fitted);

        center_shift = centers.iter().map(|&x| -1.0*(x-centers[0])).collect();
        y_shift = yshifts.iter().map(|&x| -1.0*(x-yshifts[0])).collect();
    }
    Ok((y_shift, center_shift))
}


#[pymodule]
fn txm_pal_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quadfit_mc, m)?)?;
    m.add_function(wrap_pyfunction!(gaussianfit_mc, m)?)?;
    m.add_function(wrap_pyfunction!(quadfit_single, m)?)?;
    m.add_function(wrap_pyfunction!(phase_cross_correlation_rs, m)?)?;
    m.add_function(wrap_pyfunction!(phase_cross_correlation_stack, m)?)?;
    m.add_function(wrap_pyfunction!(renormalize_absorbance_stack, m)?)?;
    m.add_function(wrap_pyfunction!(process_images, m)?)?;
    Ok(())
}
