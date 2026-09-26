#!/usr/bin/env python3
"""Export the pinned GEM-X vision stages to bounded, lossless F32 Burnpack inputs.

Python/ONNX/Torch are offline conversion and reference dependencies only.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np
import onnx
import torch
from model_export import ExportWriter, verify_file

REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
SOURCE = "32992550dba114c62243fb55e361311972dce8f9"
VIT_GRAPH = "0982dbf4f1e8a48446a6fe35329711522b60210cce2a7499ca6ab93458c87f34"
VIT_DATA = "b20ba3077ba2341d76c16dde3e58d3b37c66c3b0ec8346901a8bac14319e20fe"
SAM_CHECKPOINT = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
DINO_SOURCE = "6876159a11b4df116f30f667f8c9888617df0751"


def pin_dinov3():
    """The upstream SAM helper otherwise fetches moving DINO main via Torch Hub."""
    original = torch.hub.load
    def load(repo, *args, **kwargs):
        if repo == "facebookresearch/dinov3":
            repo = f"facebookresearch/dinov3:{DINO_SOURCE}"
        return original(repo, *args, **kwargs)
    torch.hub.load = load


def vitpose(args, writer):
    path = args.assets / "onnx/vitpose.onnx"
    verify_file(path, VIT_GRAPH)
    verify_file(path.with_suffix(".onnx.data"), VIT_DATA)
    graph = onnx.load(path, load_external_data=False).graph
    tensors = {t.name: t for t in graph.initializer}
    def get(name):
        return onnx.numpy_helper.to_array(tensors[name], base_dir=str(path.parent))
    for name in tensors:
        if name.startswith("model.backbone."):
            writer.tensor(name.removeprefix("model.backbone."), get(name))
    writer.tensor("cls_token", get("add_22"))
    angles = get("tile")
    for i in range(1, 32):
        assert np.array_equal(angles, get(f"tile_{i}"))
    writer.tensor("rope.angles", angles)
    # Linear matrices were constant-folded to [input,output] by ONNX.
    for i in range(32):
        nodes = list(graph.node)
        start = next(j for j,n in enumerate(nodes) if n.op_type == "LayerNormalization" and f"model.backbone.blocks.{i}.norm1.weight" in n.input)
        stop = next((j for j in range(start+1,len(nodes)) if nodes[j].op_type == "LayerNormalization" and any("norm1.weight" in k or k == "model.backbone.norm.weight" for k in nodes[j].input)),len(nodes))
        mats = [n for n in nodes[start:stop] if n.op_type == "MatMul" and n.input[1] in tensors]
        assert len(mats) == 5
        for n,suffix in zip(mats,["attn.qkv","attn.proj","mlp.w1","mlp.w2","mlp.w3"]):
            writer.tensor(f"blocks.{i}.{suffix}.weight", get(n.input[1]).T)
        add = next(n for n in nodes[start:stop] if n.op_type == "Add" and mats[0].output[0] in n.input)
        writer.tensor(f"blocks.{i}.attn.qkv.bias", get(add.input[1]))
    for i,j in enumerate([0,3]):
        writer.tensor(f"head.deconv.{i}.weight",get(f"model.keypoint_head.deconv_layers.{j}.weight"))
        writer.tensor(f"head.deconv.{i}.bias",get(f"model.keypoint_head.deconv_layers.{j}.weight_bias"))
    for suffix in ["weight","bias"]:
        writer.tensor(f"head.final.{suffix}",get(f"model.keypoint_head.final_layer.{suffix}"))
    return 256,192,dict(graph_sha256=VIT_GRAPH,data_sha256=VIT_DATA)


def sam_body(args,writer):
    pin_dinov3()
    sys.path.insert(0,str(args.upstream/"third_party/sam-3d-body"))
    from sam_3d_body import load_sam_3d_body
    path=args.assets/"sam3d_body.ckpt"
    checkpoint=verify_file(path,SAM_CHECKPOINT)
    model,cfg=load_sam_3d_body(str(path),device="cpu",mhr_path=str(args.mhr))
    backbone=model.backbone.float().eval()
    encoder=backbone.encoder
    state=encoder.state_dict()
    for name,value in state.items():
        if name in ["mask_token","rope_embed.periods"] or name.endswith("bias_mask"):
            continue
        if name.endswith("attn.qkv.bias"):
            value=value*state[name.removesuffix("bias")+"bias_mask"]
        writer.tensor(name,value)
    # Preserve the exact rotary convention of the independent reference.
    pe=encoder.rope_embed
    assert pe.normalize_coords=="separate"
    dd=dict(device="cpu",dtype=pe.dtype)
    coords=(torch.arange(0.5,32,**dd)/32)*2-1
    coords=torch.stack(torch.meshgrid(coords,coords,indexing="ij"),dim=-1).flatten(0,1)
    angles=(2*np.pi*coords[:,:,None]/pe.periods[None,None,:]).flatten(1,2).tile(2)
    writer.tensor("rope.angles",angles.float())
    writer.asset("SAM-LICENSE",(args.upstream/"third_party/sam-3d-body/LICENSE").read_bytes())
    # Independent full F32 Torch forward; source checkpoint tensors were BF16.
    if args.reference:
        torch.set_num_threads(8)
        backbone=backbone.cuda()
        x=torch.from_numpy(reference_input(512,512)).cuda()
        with torch.no_grad():out=backbone(x).float().cpu().numpy()
        write_reference(args.reference,x.cpu().numpy(),out)
    return 512,512,dict(checkpoint_sha256=checkpoint,reference_arithmetic="float32, original BF16 checkpoint values")


def reference_input(h,w):
    y,x=np.mgrid[:h,:w].astype(np.float32)
    rgb=np.stack([(np.sin(x/19)+1)/2,(np.cos(y/31)+1)/2,((x+y)%97)/97])
    return ((rgb-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])[None].astype(np.float32)


def write_reference(path,x,y):
    pathlib.Path(path).write_text(json.dumps(dict(revision=REVISION,shape=list(x.shape),input=x.flatten().tolist(),output_shape=list(y.shape),output=y.flatten().tolist()))+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets",type=pathlib.Path,required=True)
    ap.add_argument("--upstream",type=pathlib.Path,required=True)
    ap.add_argument("--mhr",type=pathlib.Path)
    ap.add_argument("--kind",choices=["vitpose","sam-body"],required=True)
    ap.add_argument("--out",type=pathlib.Path,required=True)
    ap.add_argument("--reference",type=pathlib.Path)
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    writer=ExportWriter(args.out)
    h,w,provenance=vitpose(args,writer) if args.kind=="vitpose" else sam_body(args,writer)
    writer.asset("LICENSE",(args.assets/"LICENSE").read_bytes())
    writer.asset("THIRD-PARTY-LICENSES",(pathlib.Path(__file__).resolve().parents[2]/"crates/burn_gem/LICENSE").read_bytes())
    writer.asset("ATTRIBUTIONS.md",(args.upstream/"ATTRIBUTIONS.md").read_bytes())
    writer.json("provenance.json",dict(revision=REVISION,source_revision=SOURCE,**provenance))
    writer.finish(model=f"nvidia/GEM-X:{args.kind}",model_revision=REVISION,source_revision=SOURCE,converter="export_gem_vision.py:v1",license="See metadata/LICENSE and upstream model-specific license notices",config=dict(kind=args.kind,height=h,width=w,layers=32,hidden=1280,heads=20,patch=16))
    if args.kind=="vitpose" and args.reference:
        import onnxruntime as ort
        session=ort.InferenceSession(str(args.assets/"onnx/vitpose.onnx"),providers=[("CUDAExecutionProvider",{"use_tf32":0}),"CPUExecutionProvider"])
        x=reference_input(h,w);y=session.run(None,{"imgs":x})[0];write_reference(args.reference,x,y)
    print(f"Exported {args.kind}: {len(writer.tensors)} tensors",flush=True)


if __name__=="__main__":main()
