//! Bounded RGB decoding, OpenCV-compatible affine crops and SOMA77 heatmaps.
use crate::camera::Crop;
use anyhow::{Result, ensure};
use image::{ImageReader, RgbImage};
use std::io::Cursor;

pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    ensure!(bytes.len() <= 32 * 1024 * 1024, "Image exceeds 32 MiB");
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode()?.to_rgb8();
    ensure!(
        u64::from(image.width()) * u64::from(image.height()) <= 16 * 1024 * 1024,
        "Image exceeds 16 megapixels"
    );
    Ok(image)
}

/// Pixel-center affine warp with constant black borders and the reference's
/// 5-bit bilinear interpolation table. ViTPose aligns 0..255; SAM aligns 0..512.
pub fn crop_rgb(image: &RgbImage, crop: Crop, sam: bool) -> Result<Vec<f32>> {
    crop.validate()?;
    let size = if sam { 512 } else { 256 };
    let width = if sam { 512 } else { 192 };
    let x_offset = if sam { 0 } else { 32 };
    let divisor = if sam { 512.0f64 } else { 255.0 };
    let step = crop.size as f64 / divisor;
    let left = crop.center[0] as f64 - crop.size as f64 / 2.0;
    let top = crop.center[1] as f64 - crop.size as f64 / 2.0;
    let mut out = vec![0.0; 3 * size * width];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for y in 0..size {
        for x in 0..width {
            // warpAffine rounds separable affine terms at 10 fractional bits first.
            let sx = (((x + x_offset) as f64 * step * 1024.0).round_ties_even() as i64
                + (left * 1024.0).round_ties_even() as i64
                + 16)
                >> 5;
            let sy = ((y as f64 * step * 1024.0 + top * 1024.0).round_ties_even() as i64 + 16) >> 5;
            let ix = sx >> 5;
            let iy = sy >> 5;
            let fx = (sx & 31) as u32;
            let fy = (sy & 31) as u32;
            for c in 0..3 {
                let mut sum = 0u32;
                for (dx, dy, w) in [
                    (0, 0, (32 - fx) * (32 - fy)),
                    (1, 0, fx * (32 - fy)),
                    (0, 1, (32 - fx) * fy),
                    (1, 1, fx * fy),
                ] {
                    let (px, py) = (ix + dx, iy + dy);
                    if px >= 0 && py >= 0 && px < image.width() as i64 && py < image.height() as i64
                    {
                        sum += image.get_pixel(px as u32, py as u32)[c] as u32 * w;
                    }
                }
                let pixel = ((sum + 512) >> 10) as f32 / 255.0;
                out[c * size * width + y * width + x] = (pixel - mean[c]) / std[c];
            }
        }
    }
    Ok(out)
}

pub fn keypoints(heatmaps: &[f32], crop: Crop, flip_test: bool) -> Result<Vec<[f32; 3]>> {
    crop.validate()?;
    let n = 77 * 64 * 48;
    ensure!(
        heatmaps.len() == n * if flip_test { 2 } else { 1 }
            && heatmaps.iter().all(|v| v.is_finite()),
        "Invalid SOMA77 heatmaps"
    );
    let mut flip: Vec<usize> = (0..77).collect();
    flip.swap(9, 10);
    for i in 11..39 {
        flip.swap(i, i + 28);
    }
    for i in 67..72 {
        flip.swap(i, i + 5);
    }
    let mut result = vec![];
    for j in 0..77 {
        let mut heat = vec![0.0f32; 64 * 48];
        for y in 0..64 {
            for x in 0..48 {
                let v = heatmaps[j * 64 * 48 + y * 48 + x];
                heat[y * 48 + x] = if flip_test {
                    (v + heatmaps[n + flip[j] * 64 * 48 + y * 48 + 47 - x]) * 0.5
                } else {
                    v
                };
            }
        }
        let mut best = 0;
        for i in 1..heat.len() {
            if heat[i] > heat[best] {
                best = i;
            }
        }
        let (ix, iy) = (best % 48, best / 48);
        let (mut x, mut y) = (ix as f32, iy as f32);
        let sign = |v: f32| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        if ix > 1 && ix < 47 {
            x += sign(heat[best + 1] - heat[best - 1]) * 0.25;
        }
        if iy > 1 && iy < 63 {
            y += sign(heat[best + 48] - heat[best - 48]) * 0.25;
        }
        result.push([
            (x / 48.0 - 0.5) * crop.size * 0.75 + crop.center[0],
            (y / 64.0 - 0.5) * crop.size + crop.center[1],
            heat[best],
        ]);
    }
    Ok(result)
}
