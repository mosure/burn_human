// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Small skeleton algebra. Large identity, corrective, and skinning operations use Burn.
use glam::{Mat3, Mat4, Quat, Vec3, Vec4};

pub fn matrix3(rows: [[f32; 3]; 3]) -> Mat3 {
    Mat3::from_cols_array_2d(&rows).transpose()
}
pub fn matrix4(rows: [[f32; 4]; 4]) -> Mat4 {
    Mat4::from_cols_array_2d(&rows).transpose()
}
pub fn rotation(m: Mat4) -> Mat3 {
    Mat3::from_mat4(m)
}
pub fn position(m: Mat4) -> Vec3 {
    m.w_axis.truncate()
}
pub fn transform(r: Mat3, t: Vec3) -> Mat4 {
    Mat4::from_cols(
        r.x_axis.extend(0.0),
        r.y_axis.extend(0.0),
        r.z_axis.extend(0.0),
        t.extend(1.0),
    )
}
pub fn local(world: &[Mat4], parents: &[usize]) -> Vec<Mat4> {
    world
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if i == parents[i] {
                *w
            } else {
                world[parents[i]].inverse() * *w
            }
        })
        .collect()
}
pub fn fk(rotations: &[Mat3], translations: &[Vec3], parents: &[usize]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; parents.len()];
    for i in 0..parents.len() {
        let m = transform(rotations[i], translations[i]);
        world[i] = if parents[i] == i {
            m
        } else {
            world[parents[i]] * m
        };
    }
    world
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}
fn row_norm(h: Mat3) -> f32 {
    let t = h.transpose();
    t.x_axis
        .abs()
        .element_sum()
        .max(t.y_axis.abs().element_sum())
        .max(t.z_axis.abs().element_sum())
}

/// SOMA's regularized Newton-Schulz alignment, with an SVD fallback.
pub fn align(target: &[Vec3], source: &[Vec3]) -> Mat3 {
    if target.is_empty() {
        return Mat3::IDENTITY;
    }
    if target.len() == 1 {
        let a = target[0].normalize_or_zero();
        let b = source[0].normalize_or_zero();
        if a == Vec3::ZERO || b == Vec3::ZERO {
            return Mat3::IDENTITY;
        }
        return Mat3::from_quat(Quat::from_rotation_arc(b, a));
    }
    let mut h = target
        .iter()
        .zip(source)
        .fold(Mat3::ZERO, |m, (&a, &b)| m + outer(a, b));
    let na = target[0].cross(target[1]);
    let nb = source[0].cross(source[1]);
    if na.length() > 1e-9 && nb.length() > 1e-9 {
        h += outer(
            na * (target[0].length() / (na.length() + 1e-8)),
            nb * (source[0].length() / (nb.length() + 1e-8)),
        );
    }
    let prior = row_norm(h).max(1e-8);
    let rank = ((1e-6 - h.determinant().abs() / prior.powi(3)) / 1e-6).clamp(0.0, 1.0);
    h += Mat3::IDENTITY * (0.05 * rank * prior);
    let mut r = h / (row_norm(h) + 1e-8);
    for _ in 0..30 {
        r = r * (Mat3::IDENTITY * 3.0 - r.transpose() * r) * 0.5;
    }
    if r.determinant() < 0.0 {
        r.z_axis = -r.z_axis;
    }
    let error = (r.transpose() * r - Mat3::IDENTITY)
        .to_cols_array()
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()));
    if !r.is_finite()
        || (r.determinant() - 1.0).abs() > 1e-2
        || error > 1e-2
        || h.determinant() < 0.0
    {
        let svd = nalgebra::Matrix3::from_column_slice(&h.to_cols_array()).svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let mut correction = nalgebra::Matrix3::identity();
        // Preserve the reference implementation's determinant convention.
        if (u * vt.transpose()).determinant() < 0.0 {
            correction[(2, 2)] = -1.0;
        }
        let output = u * correction * vt;
        r = Mat3::from_cols_array(output.as_slice().try_into().unwrap());
    }
    r
}

pub fn twist_x(mut q: Quat) -> f32 {
    q = q.normalize();
    if q.w < 0.0 {
        q = -q;
    }
    let half = Vec4::new(q.x, q.y, q.z, q.w + 1.0).normalize();
    4.0 * half.x.atan2(half.w)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_recovers_rotations_and_degenerate_cases() {
        let r = Mat3::from_quat(Quat::from_rotation_y(0.37) * Quat::from_rotation_x(-0.5));
        let source = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.3, 0.4, -0.2)];
        let target = source.map(|v| r * v * 1.3);
        let fitted = align(&target, &source);
        assert!((fitted - r).to_cols_array().iter().all(|v| v.abs() < 2e-5));
        assert!(align(&[Vec3::ZERO; 3], &[Vec3::ZERO; 3]).is_finite());
        assert!((align(&[Vec3::Y], &[Vec3::X]) * Vec3::X - Vec3::Y).length() < 1e-5);
    }
    #[test]
    fn twist_is_continuous_near_pi_and_quaternion_sign_invariant() {
        for angle in [
            -std::f32::consts::PI + 0.001,
            -1.0,
            0.0,
            1.0,
            std::f32::consts::PI - 0.001,
        ] {
            let q = Quat::from_rotation_x(angle);
            assert!((twist_x(q) - angle).abs() < 1e-5);
            assert!((twist_x(q) - twist_x(-q)).abs() < 1e-6);
        }
    }
}
