use pyo3::prelude::*;
use pyo3::types::PyDict;
use ndarray::Array2;
use geo::{Point, Polygon, Rect};
use geo::EuclideanDistance;
use geo::algorithm::contains::Contains;

pub fn create_mask(image_width: usize, image_height: usize, rois: Vec<&PyDict>) -> PyResult<Array2<u8>> {
    // 0으로 초기화된 마스크 생성
    let mut mask = Array2::<u8>::zeros((image_height, image_width));

    // ROI 처리
    for roi in rois {
        let roi_type: String = roi.get_item("type")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ROI must have a 'type' field"))?
            .extract()?;

        match roi_type.as_str() {
            "circle" => {
                // Circle: center와 radius 가져오기
                let center: (f64, f64) = roi.get_item("center")
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Circle ROI must have a 'center' field"))?
                    .extract()?;
                let radius: f64 = roi.get_item("radius")
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Circle ROI must have a 'radius' field"))?
                    .extract()?;

                let center_point = Point::new(center.0, center.1);
                for y in 0..image_height {
                    for x in 0..image_width {
                        let point = Point::new(x as f64, y as f64);
                        if point.euclidean_distance(&center_point) <= radius {
                            mask[[y, x]] = 1;
                        }
                    }
                }
            }
            "rectangle" => {
                // Rectangle: origin과 size(width, height) 가져오기
                let origin: (f64, f64) = roi.get_item("origin")
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Rectangle ROI must have an 'origin' field"))?
                    .extract()?;
                let size: (f64, f64) = roi.get_item("size")
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Rectangle ROI must have a 'size' field"))?
                    .extract()?;

                let rect = Rect::new(
                    Point::new(origin.0, origin.1),
                    Point::new(origin.0 + size.0, origin.1 + size.1),
                );
                for y in 0..image_height {
                    for x in 0..image_width {
                        let point = Point::new(x as f64, y as f64);
                        if rect.contains(&point) {
                            mask[[y, x]] = 1;
                        }
                    }
                }
            }
            "polygon" => {
                // Polygon: 다각형의 점 리스트 가져오기
                let points: Vec<(f64, f64)> = roi.get_item("points")
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Polygon ROI must have a 'points' field"))?
                    .extract()?;
                let polygon = Polygon::new(points.into(), vec![]);

                for y in 0..image_height {
                    for x in 0..image_width {
                        let point = Point::new(x as f64, y as f64);
                        if polygon.contains(&point) {
                            mask[[y, x]] = 1;
                        }
                    }
                }
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unsupported ROI type")),
        }
    }

    Ok(mask)
}
