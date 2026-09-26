#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline SOMA-X asset compiler and independent reference fixture generator.

Uses the official layer to resolve USD, RBF fitting coefficients, procedural
controls, and sparse corrective support. Runtime evaluation remains Rust/Burn.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

REVISION = "104578ed58857f6faa7592fb83d0a2dad43c36fa"
SOURCE = "cc1f3967755f8e36d187d2e26114633dbd651cd5"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=pathlib.Path, required=True)
    ap.add_argument("--assets", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.upstream))
    from soma.body.soma import SOMALayer
    torch.set_num_threads(8)
    layer = SOMALayer(data_root=args.assets, device="cpu", identity_model_type="soma", mode="torch", lod="mid")
    layer.eval()
    # Use the official Torch math as oracle, independent of any Warp kernels.
    layer.skeleton_transfer.use_warp_for_rotations = False
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "raw").mkdir()
    tensors = []
    assets = []

    def write(name, values, dtype="f32"):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype={"f32": "<f4", "i32": "<i4"}[dtype])
        raw = values.tobytes()
        filename = "raw/" + name + ".bin"
        (args.out / filename).write_bytes(raw)
        tensors.append(dict(name=name, shape=list(values.shape), dtype=dtype, file=filename,
                            sha256=hashlib.sha256(raw).hexdigest()))

    def json_asset(name, data):
        raw = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
        (args.out / name).write_bytes(raw)
        assets.append(dict(path="metadata/" + name, file=name, sha256=hashlib.sha256(raw).hexdigest()))

    def array(t):
        return t.detach().cpu().tolist()

    identity = layer.identity_model
    write("identity.mean", identity.pca_mean * identity._unit_conversion)
    write("identity.directions", identity.pca_matrix * identity._unit_conversion)
    write("identity.stddev", identity.eigenvalues.sqrt())
    fit = layer.skeleton_transfer
    write("fit.regressor", fit.sparse_rbf_matrix.to_dense())
    write("fit.vertices", fit.bind_shape)
    write("skin.weights", layer.skinning_weights)
    write("skin.public_weights", layer.public_batched_skinning.skinning_weights)
    write("mesh.faces", layer.faces, "i32")
    corrective = layer.correctives_model.module
    write("corrective.input.weight", (corrective.W1 * corrective.M1_prior).T.contiguous())
    w2 = corrective.W2 * corrective.M2_prior
    groups = []
    channels = corrective.K // 78
    for joint in range(78):
        weights = w2[joint * channels:(joint + 1) * channels]
        columns = torch.where((weights != 0).any(dim=0))[0]
        if columns.numel() == 0:
            continue
        write(f"corrective.output.{joint}.weight", weights[:, columns].T.contiguous())
        write(f"corrective.output.{joint}.indices", columns, "i32")
        groups.append(dict(joint=joint, columns=len(columns)))
    proc = layer.procedural_transforms
    procedural = {name: array(value) for name, value in proc.named_buffers()}
    full_scale_map = layer.target_to_public_joint_indices.clone()
    public_names = list(layer.public_joint_names)
    target_names = list(layer.target_joint_names)
    for segment in proc.segments:
        for twist in segment.twist_joints:
            full_scale_map[target_names.index(twist)] = public_names.index(segment.end_joint)
    meta = dict(public_names=public_names, target_names=target_names,
        public_parents=fit.joint_parent_ids, target_parents=array(layer.joint_parent_ids),
        public_indices=array(layer.public_transform_joint_indices),
        bind_world=array(layer.bind_pose_world), bind_local=array(layer.bind_pose_local),
        fit_bind_world=array(fit.bind_world_transforms), fit_bind_local=array(fit.bind_local_transforms),
        fit_skinned=[array(torch.where(fit.skinning_weights[:, i] > 0.01)[0]) for i in range(78)],
        fit_frozen=sorted(fit.freeze_rotations), fit_skip_endjoints=fit.skip_endjoints,
        fit_skip_inverse_lbs=fit.skip_inverse_lbs, procedural=procedural,
        corrective_bind=array(corrective.bindpose), corrective_tanh=corrective.use_tanh,
        corrective_channels=channels, corrective_groups=groups,
        scale_names=list(layer.scale_param_names), scale_public_indices=array(layer.bone_scale_public_joint_indices),
        scale_target_map=array(full_scale_map))
    json_asset("rig.json", meta)
    source_hashes = {}
    for filename in ["SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json", "correctives_model.pt"]:
        with (args.assets / filename).open("rb") as f:
            source_hashes[filename] = hashlib.file_digest(f, "sha256").hexdigest()
    json_asset("provenance.json", dict(source_revision=SOURCE, asset_revision=REVISION, hashes=source_hashes,
        identity="SOMA PCA", unit="meters", lod="mid", rotation_fit="auto", procedural_mode=proc.mode,
        correctives="exact sparse support, F32 values, no threshold pruning"))
    license_raw = (args.assets / "LICENSE").read_bytes()
    (args.out / "LICENSE").write_bytes(license_raw)
    assets.append(dict(path="metadata/LICENSE", file="LICENSE", sha256=hashlib.sha256(license_raw).hexdigest()))
    export = dict(model="nvidia/SOMA-X", model_revision=REVISION, source_revision=SOURCE,
        converter="export_soma.py:v1", license="Apache-2.0; see metadata/LICENSE",
        config=dict(vertices=18056, public_joints=78, skin_joints=110, identity_coefficients=128,
                    scale_parameters=60, lod="mid", unit="meters", up_axis="Y", forward_axis="Z"),
        tensors=tensors, assets=assets)
    (args.out / "export.json").write_text(json.dumps(export, indent=2) + "\n")
    cases = []
    with torch.no_grad():
        for case in range(4):
            coefficients = torch.zeros(1, 128)
            if case:
                coefficients[0, :8] = torch.tensor([0.4, -0.7, 0.2, 0.1, -0.3, 0.5, 0.2, -0.1]) * case / 2
            scales = torch.ones(1, layer.num_scale_params)
            if case >= 2:
                scales[0, :3] = torch.tensor([1.08, 0.93, 1.02])
                scales[0, -4:] = torch.tensor([1.04, 0.97, 1.03, 0.98])
            global_scale = [1.0, 0.9, 1.1, 1.0][case]
            layer.prepare_identity(coefficients, scales, global_scale=global_scale)
            rotations = torch.zeros(1, 77, 3)
            if case:
                for name, r in [("Hips", [0.1, 0.4, -0.1]), ("LeftArm", [0.3, 0.2, 0.5]),
                                ("LeftForeArm", [0.5, 0.1, 0.8]), ("RightArm", [-0.2, 0.1, -0.3]),
                                ("LeftHandIndex2", [0.1, 0.05, 0.3]), ("LeftLeg", [0.2, 0.05, 0.0])]:
                    rotations[0, public_names.index(name) - 1] = torch.tensor(r)
            translation = torch.tensor([[0.2, 1.1, -0.3]])
            posed = layer.pose(rotations, transl=translation, apply_correctives=case != 3)
            cases.append(dict(name=["neutral", "shape_pose", "scales_correctives", "no_correctives"][case],
                coefficients=array(coefficients[0]), scales=array(scales[0]), global_scale=global_scale,
                rotations=array(rotations[0]), translation=array(translation[0]), correctives=case != 3,
                rest_vertices=array(layer._cached_rest_shape[0]), bind_world=array(layer._cached_bind_transforms_world[0]),
                vertices=array(posed["vertices"][0]), transforms=array(posed["transforms"][0])))
    (args.out / "reference.json").write_text(json.dumps(dict(source_revision=SOURCE, revision=REVISION, cases=cases)) + "\n")
    print(f"Exported {len(tensors)} tensors, {len(groups)} sparse corrective groups, {len(cases)} reference cases", flush=True)


if __name__ == "__main__":
    main()
