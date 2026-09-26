#!/usr/bin/env python3
"""Lossless GEM-X regression/denoising export and released ONNX parity fixtures."""
import argparse
import importlib.util
import json
import pathlib
import torch
import numpy as np
import onnx
import onnxruntime as ort
from model_export import ExportWriter,verify_file
from export_gem_vision import REVISION,SOURCE

CHECKPOINT="4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
GRAPH="20aab83c01bbd909a258ad0fa465458eda382c63dc80d56952b2ec952d6e192c"
DATA="720c2400281b730574bf2a446c33ecd77bf494ac8161b081b09bb122b9fad2f8"

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--assets",type=pathlib.Path,required=True);ap.add_argument("--upstream",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args()
    verify_file(args.assets/"gem_soma.ckpt",CHECKPOINT);verify_file(args.assets/"onnx/gem_denoiser.onnx",GRAPH);verify_file(args.assets/"onnx/gem_denoiser.onnx.data",DATA)
    state=torch.load(args.assets/"gem_soma.ckpt",map_location="cpu",weights_only=False)["state_dict"]
    writer=ExportWriter(args.out)
    for k,v in state.items():
        k=k.removeprefix("pipeline.denoiser3d.denoiser.")
        if k.startswith("static_conf_head") or k=="embed_timestep.sequence_pos_encoder.pe":continue
        if k=="sequence_pos_encoder.pe":k="time.positions";v=v[:1000,0,:]
        writer.tensor(k,v)
    spec=importlib.util.spec_from_file_location("stats",args.upstream/"gem/network/stats_compose.py");stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    stats=stats.MM_V2_SOMA_METROSIM
    graph=onnx.load(args.assets/"onnx/gem_denoiser.onnx",load_external_data=False);tensors={t.name:t for t in graph.graph.initializer}
    def value(name):return onnx.numpy_helper.to_array(tensors[name],base_dir=str(args.assets/"onnx"))
    assert len(stats["mean"])==585
    writer.json("decode.json",dict(mean=np.asarray(stats["mean"]).tolist(),std=np.maximum(1,np.asarray(stats["std"])).tolist(),observation_joints=value("val_88").flatten().tolist()))
    writer.asset("LICENSE",(args.assets/"LICENSE").read_bytes());writer.json("provenance.json",dict(revision=REVISION,source_revision=SOURCE,checkpoint_sha256=CHECKPOINT,graph_sha256=GRAPH,data_sha256=DATA,reference="Released fast inference path: zero latent, timestep 999; full SAM feature"))
    writer.finish(model="nvidia/GEM-X:denoiser",model_revision=REVISION,source_revision=SOURCE,converter="export_gem_denoiser.py:v1",license="Apache-2.0; see metadata/LICENSE",config=dict(layers=12,hidden=512,heads=8,features=585,observation_joints=33,image_features=1024,scale_components=28,max_frames=120))
    session=ort.InferenceSession(str(args.assets/"onnx/gem_denoiser.onnx"),providers=[("CUDAExecutionProvider",{"use_tf32":0}),"CPUExecutionProvider"])
    gen=np.random.default_rng(618);cases=[]
    for batch,length in [(1,1),(2,8)]:
        obs=gen.uniform([30,50,0],[600,450,1],size=(batch,length,77,3)).astype(np.float32)
        box=np.tile(np.array([320,240,400],np.float32),(batch,length,1));k=np.tile(np.array([[800,0,320],[0,800,240],[0,0,1]],np.float32),(batch,length,1,1))
        image=gen.normal(0,.5,(batch,length,1024)).astype(np.float32);cam=np.tile(np.array([1,0,0,0,1,0],np.float32),(batch,length,1))
        values=dict(obs=obs,bbx_xys=box,K_fullimg=k,f_imgseq=image,f_cam_angvel=cam)
        x,c=session.run(None,values);cases.append(dict(batch=batch,frames=length,observations=obs.flatten().tolist(),boxes=box.reshape(-1,3).tolist(),cameras=k.reshape(-1,3,3).tolist(),features=image.flatten().tolist(),angular=cam.reshape(-1,6).tolist(),prediction=x.flatten().tolist(),camera=c.flatten().tolist()))
    (args.out/"reference.json").write_text(json.dumps(dict(revision=REVISION,cases=cases))+"\n");print(f"Exported GEM {len(writer.tensors)} tensors",flush=True)

if __name__=="__main__":main()
