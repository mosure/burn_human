#!/usr/bin/env python3
"""Convert SOMA-X animation NPZ to the portable SomaAnimation JSON contract.

Supply the animation's identity-fitted bind rig as RigDefinition JSON in metres.
This converter does not synthesize a SOMA rig from ARDY Core or evaluate a SOMA
identity/corrective model. Identity metadata survives the conversion for future
body backends. Requires numpy and scipy; never enables NumPy pickle loading.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation


def rotations(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid rotation matrix shape or values")
    if not np.allclose(matrix @ np.swapaxes(matrix, -2, -1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(matrix), 1, atol=1e-4):
        raise ValueError("Rotation matrices must be in SO(3)")
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3))


def single_identity(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        if not np.all(values == values[:1]):
            raise ValueError("Per-frame identity changes need separate fitted rigs/clips")
        values = values[0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid identity/scale coefficients")
    return values.tolist()


def convert(data, rig, fps, revision):
    names = list(map(str, data["joint_names"]))
    poses = np.asarray(data["poses"], dtype=np.float32)
    if str(data["rotation_repr"]) == "matrix":
        poses = rotations(poses).as_rotvec().reshape(*poses.shape[:2], 3)
    elif str(data["rotation_repr"]) != "rotvec":
        raise ValueError("Unsupported SOMA rotation representation")
    if poses.ndim != 3 or poses.shape[1:] != (len(names), 3) or not np.isfinite(poses).all():
        raise ValueError("Invalid pose array")
    if bool(data.get("keep_root", False)):
        if names[0] != "Root" or not np.allclose(poses[:, 0], 0, atol=1e-6):
            raise ValueError("Only an identity virtual Root can be removed")
        names, poses = names[1:], poses[:, 1:]
    if len(names) not in [30, 77] or rig["id"] != f"soma-{len(names)}" or names != [j["name"] for j in rig["joints"]]:
        raise ValueError("Supply the matching SOMA-30/77 bind rig in exact joint order")
    scale = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001, "m": 1.0, "cm": 0.01, "mm": 0.001}[str(data["unit"])]
    translation = np.asarray(data["transl"], dtype=np.float32) * scale
    if translation.shape != (len(poses), 3) or not np.isfinite(translation).all():
        raise ValueError("Invalid translation array")
    orient = None
    if not bool(data["absolute_pose"]):
        orient = np.asarray(data["joint_orient"])
        if orient.shape == (len(names) + 1, 3, 3):
            if not np.allclose(orient[0], np.eye(3), atol=1e-4):
                raise ValueError("Virtual Root orientation must be identity")
            orient = orient[1:]
        if orient.shape != (len(names), 3, 3):
            raise ValueError("joint_orient must match the supplied rig")
        orient = rotations(orient).as_quat().tolist()
    if not 0 < fps <= 1000:
        raise ValueError("Invalid frame rate")
    return dict(rig=rig, fps=fps, identity_model_type=str(data["identity_model_type"]),
                identity_coeffs=single_identity(data["identity_coeffs"]),
                scale_params=single_identity(data.get("scale_params", np.zeros(0))),
                global_scale=float(data["global_scale"]) if "global_scale" in data else None,
                poses=poses.tolist(), translations=translation.tolist(),
                absolute_pose=bool(data["absolute_pose"]), joint_orient=orient,
                unit="meters", source_revision=revision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animation", type=pathlib.Path, required=True)
    ap.add_argument("--rig", type=pathlib.Path, required=True, help="Identity-fitted RigDefinition JSON, metres")
    ap.add_argument("--fps", type=float, required=True, help="SOMA NPZ does not prescribe a frame rate")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rig = json.loads(args.rig.read_text())
    digest = hashlib.sha256(args.animation.read_bytes()).hexdigest()
    with np.load(args.animation, allow_pickle=False) as data:
        result = convert(data, rig, args.fps, f"npz-sha256:{digest}")
    with args.out.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
