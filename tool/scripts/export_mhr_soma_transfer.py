#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compile MHR-to-SOMA tetrahedral transfer and the fixed facial Laplacian solve."""
import argparse
import json
import pathlib
import sys
import torch
from model_export import ExportWriter
from export_soma import REVISION, SOURCE
from export_mhr import CHECKPOINT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=pathlib.Path, required=True)
    ap.add_argument("--assets", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.upstream))
    from soma.body.soma import SOMALayer
    torch.set_num_threads(8)
    layer = SOMALayer(data_root=args.assets, device="cpu", identity_model_type="mhr", mode="torch", lod="mid")
    model = layer.identity_model
    interp = model._to_soma_interp
    lap = model._laplacian_mesh
    assert interp.tet_normal_scale == "area" and lap.constraint_mode == "hard" and lap.jitter == 0
    writer = ExportWriter(args.out)
    faces = interp.F_src_torch[interp.face_ids]
    for i in range(3):
        writer.tensor(f"transfer.source.{i}", faces[:, i], "i32")
    writer.tensor("transfer.barycentric", interp.bary_coords)
    lfg = lap.L_FG.to_dense()
    columns = torch.where((lfg != 0).any(dim=0))[0]
    boundary = lap.vid_constrained[columns]
    # Factorization is identity-independent. Compile its fixed linear operator
    # without replacing or approximating the live shape-dependent right hand side.
    matrix = torch.cholesky_solve(-lap._hard_chol_sign * lfg[:, columns], lap._chol_factor)
    bias = torch.cholesky_solve(lap._hard_chol_sign * lap.btilde, lap._chol_factor)
    writer.tensor("transfer.boundary", boundary, "i32")
    writer.tensor("transfer.unknown", lap.vid_unknown, "i32")
    writer.tensor("transfer.solve", matrix)
    writer.tensor("transfer.bias", bias)
    writer.asset("LICENSE", (args.assets / "LICENSE").read_bytes())
    writer.json("provenance.json", dict(source_revision=SOURCE, revision=REVISION, mhr_sha256=CHECKPOINT,
        operation="MHR tetrahedral transfer plus exact fixed hard-constraint Laplacian operator", native_unit="centimeters", output_unit="meters"))
    writer.finish(model="nvidia/SOMA-X:MHR-transfer", model_revision=REVISION, source_revision=SOURCE,
        converter="export_mhr_soma_transfer.py:v1", license="Apache-2.0; see metadata/LICENSE",
        config=dict(source_vertices=18439, target_vertices=18056, boundary=len(boundary), unknown=len(lap.vid_unknown), normal_scale="area", output_unit="meters"))
    gen = torch.Generator().manual_seed(834)
    cases = []
    with torch.no_grad():
        for i in range(3):
            identity = torch.randn((1,45),generator=gen) * 0.3 * i
            scales = torch.randn((1,68),generator=gen) * 0.08 * i
            flex = torch.randn((1,6),generator=gen) * 0.03 * i
            shape = model(identity, scales, kwargs={"bone_length_flexibles":flex})
            cases.append(dict(identity=identity[0].tolist(),scales=scales[0].tolist(),flex=flex[0].tolist(),vertices=shape[0].tolist()))
    (args.out / "reference.json").write_text(json.dumps(dict(revision=REVISION,cases=cases)) + "\n")
    print(f"Compiled {len(lap.vid_unknown)} unknown vertices from {len(boundary)} boundary vertices",flush=True)


if __name__ == "__main__":
    main()
