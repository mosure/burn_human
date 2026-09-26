//! Continuous pose parameters to MHR's Euler/scale parameterization.
//! Small skeleton algebra stays on the host; learned projections and meshes stay on Burn.
use anyhow::{Result, ensure};
use burn_mhr::MhrInput;
use glam::{EulerRot, Mat3, Vec3};
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Head {
    pub scale_mean: Vec<f32>,
    pub scale_comps: Vec<Vec<f32>>,
    pub hand_mean: Vec<f32>,
    pub hand_left: Vec<usize>,
    pub hand_right: Vec<usize>,
    pub body_triples: Vec<[usize; 3]>,
    pub body_single: Vec<usize>,
}
fn normalize(v: Vec3) -> Vec3 {
    v / v.length().max(1e-12)
}
pub(crate) fn rotation6(v: &[f32]) -> Mat3 {
    let x = normalize(Vec3::from_slice(&v[..3]));
    let a = Vec3::from_slice(&v[3..6]);
    let y = normalize(a - x * x.dot(a));
    Mat3::from_cols(x, y, x.cross(y))
}
fn xyz(v: &[f32]) -> [f32; 3] {
    let x = normalize(Vec3::from_slice(&v[..3]));
    let z = normalize(x.cross(Vec3::from_slice(&v[3..6])));
    let y = z.cross(x);
    let sy = (x.x * x.x + x.y * x.y).sqrt();
    if sy < 1e-6 {
        [(-z.y).atan2(y.y), (-x.z).atan2(sy), 0.0]
    } else {
        [y.z.atan2(z.z), (-x.z).atan2(sy), x.y.atan2(x.x)]
    }
}
impl Head {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.scale_mean.len() == 68
                && self.scale_comps.len() == 28
                && self.scale_comps.iter().all(|v| v.len() == 68)
                && self.hand_mean.len() == 54
                && self.hand_left.len() == 27
                && self.hand_right.len() == 27
                && self
                    .hand_left
                    .iter()
                    .chain(&self.hand_right)
                    .all(|x| *x < 136)
                && self.body_triples.len() == 23
                && self.body_single.len() == 58
                && self
                    .body_triples
                    .iter()
                    .flatten()
                    .chain(&self.body_single)
                    .all(|i| *i < 133),
            "Invalid SAM head metadata"
        );
        ensure!(
            self.scale_mean
                .iter()
                .chain(self.scale_comps.iter().flatten())
                .chain(&self.hand_mean)
                .all(|x| x.is_finite()),
            "Non-finite SAM metadata"
        );
        Ok(())
    }
    pub fn parameters(&self, pred: &[f32]) -> Result<MhrInput> {
        ensure!(
            pred.len() == 519 && pred.iter().all(|x| x.is_finite()),
            "Invalid SAM head prediction"
        );
        let mut result = MhrInput {
            identity: pred[266..311].to_vec(),
            ..Default::default()
        };
        let (z, y, x) = rotation6(&pred[..6]).to_euler(EulerRot::ZYX);
        result.parameters[3..6].copy_from_slice(&[z, y, x]);
        let cont = &pred[6..266];
        let mut body = [0.0f32; 133];
        for (ids, cont) in self
            .body_triples
            .iter()
            .zip(cont[..138].as_chunks::<6>().0.iter())
        {
            for (i, v) in ids.iter().zip(xyz(cont)) {
                body[*i] = v;
            }
        }
        for (idx, v) in self
            .body_single
            .iter()
            .zip(cont[138..254].as_chunks::<2>().0.iter())
        {
            body[*idx] = v[0].atan2(v[1]);
        }
        body[124..130].copy_from_slice(&cont[254..260]);
        body[62..116].fill(0.0);
        body[130..].fill(0.0);
        result.parameters[6..136].copy_from_slice(&body[..130]);
        for (hand, indices) in [
            (&pred[339..393], &self.hand_left),
            (&pred[393..447], &self.hand_right),
        ] {
            let cont: Vec<f32> = hand
                .iter()
                .zip(&self.hand_mean)
                .map(|(a, b)| a + b)
                .collect();
            let mut cursor = 0;
            let mut p = vec![];
            for dof in [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1] {
                if dof == 3 {
                    p.extend(xyz(&cont[cursor..cursor + 6]));
                } else {
                    for i in 0..dof {
                        p.push(cont[cursor + 2 * i].atan2(cont[cursor + 2 * i + 1]));
                    }
                }
                cursor += 2 * dof;
            }
            for (index, value) in indices.iter().zip(p) {
                result.parameters[*index] = value;
            }
        }
        for j in 0..68 {
            result.parameters[136 + j] = self.scale_mean[j]
                + (0..28)
                    .map(|i| pred[311 + i] * self.scale_comps[i][j])
                    .sum::<f32>();
        }
        result.validate()?;
        Ok(result)
    }
}
