use ndarray::prelude::*;
use ndrustfft::{ndfft, ndfft_r2c, ndifft};
use ndrustfft::{FftHandler, R2cFftHandler};
use num_complex::Complex;

#[allow(dead_code)]
fn rfft2_to_fft2(im_shape: (usize, usize), rfft: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let fcols = im_shape.1;
    let fft_cols = rfft.shape()[1];

    let mut result = Array2::zeros(im_shape);

    result.slice_mut(s![.., ..fft_cols]).assign(&rfft);

    let top = rfft.slice(s![0, 1..]).to_owned();
    if fcols % 2 == 0 {
        result
            .slice_mut(s![0, fft_cols - 1..])
            .assign(&top.slice(s![..;-1]).mapv(|c| c.conj()));
        let mid = rfft.slice(s![1.., 1..]).to_owned();
        let mid_conj = mid.slice(s![..;-1, ..;-1]).mapv(|c| c.conj());
        result
            .slice_mut(s![1.., 1..mid.shape()[1] + 1])
            .assign(&mid);
        result
            .slice_mut(s![1.., mid.shape()[1] + 1..])
            .assign(&mid_conj.slice(s![.., 1..]));
    } else {
        result
            .slice_mut(s![0, fft_cols..])
            .assign(&top.slice(s![..;-1]).mapv(|c| c.conj()));
        let mid = rfft.slice(s![1.., 1..]).to_owned();
        let mid_conj = mid.slice(s![..;-1, ..;-1]).mapv(|c| c.conj());
        result
            .slice_mut(s![1.., 1..mid.shape()[1] + 1])
            .assign(&mid);
        result
            .slice_mut(s![1.., mid.shape()[1] + 1..])
            .assign(&mid_conj);
    }

    result
}

#[allow(dead_code)]
pub fn rfftn(data: &Array2<f64>, rows: usize, cols: usize) -> Array2<Complex<f64>> {
    // Init arrays
    let vhat_cols = cols / 2 + 1;
    let mut vhat: Array2<Complex<f64>> = Array2::zeros((rows, vhat_cols));

    // Init handlers
    let mut handler_ax0 = FftHandler::<f64>::new(rows);
    let mut handler_ax1 = R2cFftHandler::<f64>::new(cols);

    // Transform
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((rows, vhat_cols));
        ndfft_r2c(&data, &mut work, &mut handler_ax1, 1);
        ndfft(&work, &mut vhat, &mut handler_ax0, 0);
    }

    let full_fft = rfft2_to_fft2((rows, cols), vhat);
    full_fft
}

pub fn fftn(data: &Array2<f64>) -> Array2<Complex<f64>> {
    // Init arrays
    let (rows, cols) = data.dim();
    let mut vhat: Array2<Complex<f64>> = Array2::zeros((rows, cols));

    // Convert to complex
    let data = data.mapv(|v| Complex::new(v, 0.0));

    // Init handlers
    let mut handler_ax0 = FftHandler::<f64>::new(rows);
    let mut handler_ax1 = FftHandler::<f64>::new(cols);

    // Transform
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((rows, cols));
        ndfft(&data, &mut work, &mut handler_ax1, 1);
        ndfft(&work, &mut vhat, &mut handler_ax0, 0);
    }
    vhat
}

pub fn ifftn(data: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    // Init arrays
    let (rows, cols) = data.dim();
    let mut vhat: Array2<Complex<f64>> = Array2::zeros((rows, cols));

    // Init handlers
    let mut handler_ax0 = FftHandler::<f64>::new(rows);
    let mut handler_ax1 = FftHandler::<f64>::new(cols);

    // Transform
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((rows, cols));
        ndifft(&data, &mut work, &mut handler_ax0, 0);
        ndifft(&work, &mut vhat, &mut handler_ax1, 1);
    }
    vhat
}

#[allow(dead_code)]
pub fn fftshift(data: &mut Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (rows, cols) = data.dim();
    let (mid_row, mid_col) = (rows / 2, cols / 2);

    // fftshift along rows
    for row in 0..mid_row {
        data.swap((row, 0), (row + mid_row, 0));
    }

    // fftshift along cols
    for col in 0..mid_col {
        data.swap((0, col), (0, col + mid_col));
    }

    data.clone()
}

pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let mut result = Vec::with_capacity(n);
    let m = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };

    for i in 0..m {
        result.push(i as f64 * val);
    }

    let n_size = n as isize;
    for i in (if n % 2 == 0 {
        -(n_size / 2)
    } else {
        -(n_size / 2)
    })..0
    {
        result.push(i as f64 * val);
    }

    result
}

pub fn phase_cross_correlation(
    reference_image: &Array2<f64>,
    moving_image: &Array2<f64>,
    upsample_factor: usize,
) -> (f64, f64) {
    // Calculate Cross Correlation
    let ref_image_fft = fftn(&reference_image);
    let moving_image_fft = fftn(&moving_image);
    let image_product = ref_image_fft * moving_image_fft.mapv(|x| x.conj());

    // Normalization
    // let eps: f64 = std::f64::EPSILON;
    // image_product.mapv_inplace(|c| c / (c.norm().max(eps * 100.)));

    let cross_correlation = ifftn(&image_product);

    // Locate maximum index
    let first_norm = cross_correlation[[0, 0]].norm();
    let mut max_index =
        cross_correlation
            .indexed_iter()
            .fold((0.0, 0.0, first_norm), |acc, (index, &value)| {
                if value.norm() > acc.2 {
                    (index.0 as f64, index.1 as f64, value.norm())
                } else {
                    acc
                }
            });

    // Midpoint
    let midpoint: (f64, f64) = (
        (reference_image.shape()[0] as f64 / 2.0).trunc(),
        (reference_image.shape()[1] as f64 / 2.0).trunc(),
    );

    // Substract max_index if it is greater than midpoint
    if max_index.0 > midpoint.0 {
        max_index.0 -= reference_image.shape()[0] as f64;
    }

    if max_index.1 > midpoint.1 {
        max_index.1 -= reference_image.shape()[1] as f64;
    }

    // Upsampled dft
    let upsample_factor: f64 = upsample_factor as f64;

    // Get shift within 1 pixel precision
    let mut shift: (f64, f64) = (
        (max_index.0 * upsample_factor).round() / upsample_factor,
        (max_index.1 * upsample_factor).round() / upsample_factor,
    );

    let upsampled_region_size: f64 = (upsample_factor * 1.5).ceil();
    let dftshift: f64 = (upsampled_region_size / 2.0).trunc();

    // Matrix multiply DFT around the current shift estimate
    let sample_region_offset: (f64, f64) = (
        -1. * shift.0 * upsample_factor + dftshift,
        -1. * shift.1 * upsample_factor + dftshift,
    );

    // Define the imaginary unit
    let im2pi = Complex::new(0.0, 2.0 * std::f64::consts::PI);

    // Define the properties of the dimensions
    let dim_properties: Vec<(usize, f64, f64)> = vec![
        (
            reference_image.shape()[0],
            upsampled_region_size,
            sample_region_offset.0,
        ),
        (
            reference_image.shape()[1],
            upsampled_region_size,
            sample_region_offset.1,
        ),
    ];

    // Upsampling
    let mut upsampled_data: Array2<Complex<f64>> = image_product.clone().mapv(|x| x.conj());

    for (n_items, ups_size, ax_offset) in dim_properties.iter().rev() {
        let mut kernel: Array2<f64> = Array::zeros((*ups_size as usize, *n_items));
        let fft_freq = fftfreq(*n_items, upsample_factor);
        for (i, mut row) in kernel.outer_iter_mut().enumerate() {
            for (j, freq) in fft_freq.iter().enumerate() {
                let x = (i as f64 - ax_offset) * freq;
                row[j] = x;
            }
        }

        let kernel = kernel.mapv(|x| (-im2pi * x).exp());

        // Perform the tensor dot product
        upsampled_data = kernel.dot(&upsampled_data.t().view());
    }

    // Calculate the shift
    let cross_correlation = upsampled_data.mapv(|x| x.conj());

    // Locate maximum index
    let first_norm = cross_correlation[[0, 0]].norm();
    let max_index =
        cross_correlation
            .indexed_iter()
            .fold((0.0, 0.0, first_norm), |acc, (index, &value)| {
                if value.norm() > acc.2 {
                    (index.0 as f64, index.1 as f64, value.norm())
                } else {
                    acc
                }
            });

    let maxima = (max_index.0 - dftshift, max_index.1 - dftshift);
    shift = (
        shift.0 + maxima.0 / upsample_factor,
        shift.1 + maxima.1 / upsample_factor,
    );

    // Return result
    shift
}

