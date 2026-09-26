#!/usr/bin/env python3
"""Compile the pinned MHR TorchScript checkpoint into bounded Burnpack input.

Exports exact F32 values; the large corrective projection is split by output
rows. The independent TorchScript oracle exercises identity, scale, expression,
articulation, skinning and correctives together.
"""
import argparse
import json
import pathlib
import torch
from model_export import ExportWriter, verify_file

REVISION = "104578ed58857f6faa7592fb83d0a2dad43c36fa"
CHECKPOINT = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    path = args.assets / "MHR/mhr_model_lod1.pt"
    verify_file(path, CHECKPOINT)
    torch.set_num_threads(8)
    model = torch.jit.load(str(path), map_location="cpu").eval()
    character = model.character_torch
    w = ExportWriter(args.output)
    w.tensor("identity.mean", character.blend_shape.base_shape.reshape(-1))
    w.tensor("identity.directions", character.blend_shape.shape_vectors.reshape(45, -1).T)
    w.tensor("expression.directions", model.face_expressions_model.shape_vectors.reshape(72, -1).T)
    p = character.parameter_transform.parameter_transform
    assert torch.count_nonzero(p[:, 204:]) == 0
    w.tensor("joint.parameters", p[:, :204])
    skin = character.linear_blend_skinning
    weights = torch.zeros(18439, 127)
    weights.index_put_((skin.vert_indices_flattened.long(), skin.skin_indices_flattened.long()),
                       skin.skin_weights_flattened, accumulate=True)
    w.tensor("skin.weights", weights)
    w.tensor("mesh.faces", character.mesh.faces, "i32")
    predictor = model.pose_correctives_model.pose_dirs_predictor
    first = getattr(predictor, "0")
    w1 = torch.sparse_coo_tensor(first.sparse_indices, first.sparse_weight, first.sparse_shape).to_dense()
    assert w1.shape == (3000, 750)
    w.tensor("corrective.input.weight", w1)
    w2 = getattr(predictor, "2").weight
    assert w2.shape == (55317, 3000)
    assert getattr(predictor, "2").bias is None
    for index, start in enumerate(range(0, 55317, 4096)):
        w.tensor(f"corrective.output.{index}.weight", w2[start:start + 4096])
    skel = character.skeleton
    rig = dict(parents=skel.joint_parents.tolist(), translations=skel.joint_translation_offsets.tolist(),
               prerotations=skel.joint_prerotations.tolist(), inverse_bind=skin.inverse_bind_pose.tolist())
    w.json("rig.json", rig)
    w.json("provenance.json", dict(source="nvidia/SOMA-X/MHR/mhr_model_lod1.pt", revision=REVISION,
        checkpoint_sha256=CHECKPOINT, upstream="https://github.com/facebookresearch/MHR", unit="centimeters"))
    w.asset("LICENSE", (args.assets / "LICENSE").read_bytes())
    w.finish(model="facebookresearch/MHR:SOMA-X-lod1", model_revision=REVISION,
        source_revision=REVISION, converter="export_mhr.py:v1", license="Apache-2.0; see metadata/LICENSE",
        config=dict(vertices=18439, joints=127, identity=45, expression=72, parameters=204,
                    corrective_hidden=3000, corrective_rows_per_object=4096, unit="centimeters"))
    generator = torch.Generator().manual_seed(3719)
    cases = []
    with torch.no_grad():
        for i in range(4):
            identity = torch.zeros(1, 45) if i == 0 else torch.randn((1, 45), generator=generator) * 0.35
            params = torch.zeros(1, 204)
            if i:
                params[:, 136:] = torch.randn((1, 68), generator=generator) * 0.08
            if i >= 2:
                params[:, 6:130] = torch.randn((1, 124), generator=generator) * 0.15
            expressions = torch.zeros(1, 72)
            if i == 3:
                expressions = torch.randn((1, 72), generator=generator) * 0.2
            vertices, state = model(identity, params, expressions, i != 3)
            cases.append(dict(identity=identity[0].tolist(), parameters=params[0].tolist(),
                expression=expressions[0].tolist(), correctives=i != 3,
                vertices=vertices[0].tolist(), skeleton=state[0].tolist()))
    (args.output / "reference.json").write_text(json.dumps(dict(checkpoint=CHECKPOINT,cases=cases)) + "\n")
    print(f"Exported {len(w.tensors)} MHR tensors and {len(cases)} independent reference cases", flush=True)


if __name__ == "__main__":
    main()
