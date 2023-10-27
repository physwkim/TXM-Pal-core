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
            for j in 0..radius+1 {
                filter.consume(arr[j]);
            }
            filtered[i] = filter.median();

        // right edge handling
        } else if i >= arr.len().saturating_sub(radius) {
            if edge_mode == "extension" {
                filter.consume(arr[arr.len()-1]);
            } else if edge_mode ==  "zeropadding" {
                filter.consume(0.0);
            }
        }
        else {
            filtered[i] = filter.consume(arr[i+radius]);
        }
    }
    filtered
}
