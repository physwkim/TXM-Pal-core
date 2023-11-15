use median::Filter;

pub fn medfilt(arr: Vec<f64>, window_size: usize, edge_mode: &str) -> Vec<f64> {
    let radius = window_size / 2;
    let mut filtered: Vec<f64> = vec![0.0; arr.len()];
    let mut filter = Filter::new(window_size);

    for i in 0..arr.len() {
        // left edge handling
        if i == 0 {
            // insert arr[0] into filter when (i-radius) < 0
            for _ in 0..radius {
                if edge_mode == "extension" {
                    filter.consume(arr[0]);
                } else if edge_mode == "zeropadding" {
                    filter.consume(0.0);
                }
            }

            // insert arr[0..radius+1] into filter
            for j in 0..radius + 1 {
                filter.consume(arr[j]);
            }
            filtered[i] = filter.median();

        // right edge handling
        } else if i >= arr.len().saturating_sub(radius) {
            if edge_mode == "extension" {
                filter.consume(arr[arr.len() - 1]);
            } else if edge_mode == "zeropadding" {
                filter.consume(0.0);
            }
        } else {
            filtered[i] = filter.consume(arr[i + radius]);
        }
    }
    filtered
}

pub fn multi_3point_average(arr: &Vec<f64>, iteration: usize) -> Vec<f64> {
    // Do 3-point convolution filter
    let one_third: f64 = 1.0 / 3.0;
    let mut filtered: Vec<f64> = arr.clone();
    let mut buffer: Vec<f64> = arr.clone();

    for _ in 0..iteration {
        for i in 0..filtered.len() {
            if i == 0 {
                filtered[i] = one_third * (buffer[i] + buffer[i + 1]);
            } else if i == arr.len() - 1 {
                filtered[i] = one_third * (buffer[i - 1] + buffer[i]);
            } else {
                filtered[i] = one_third * (buffer[i - 1] + buffer[i] + buffer[i + 1]);
            }
        }
        buffer = filtered.clone();
    }
    filtered
}

pub fn boxcar(arr: &Vec<f64>, kernel_size: usize) -> Vec<f64> {
    // Do boxcar filter
    let kernel_weight: f64 = 1.0 / kernel_size as f64;
    let mut filtered: Vec<f64> = arr.clone();
    let half_kernel: usize = kernel_size / 2;

    for i in 0..filtered.len() {
        let start = if i >= half_kernel { i - half_kernel } else { 0 };
        let end = usize::min(i + half_kernel + 1, filtered.len());
        filtered[i] = arr[start..end].iter().sum::<f64>() * kernel_weight;
    }
    filtered
}