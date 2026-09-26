#!/usr/bin/env python3
"""Export pinned ARDY metadata and real-checkpoint parity fixtures.

Run in an ARDY environment with PYTHONPATH pointing at its pinned source tree.
This script never downloads a gated text encoder. Random embeddings test numeric
parity only; they are not evidence of semantic prompt quality.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np
import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=pathlib.Path, required=True)
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    import ardy
    if not pathlib.Path(ardy.__file__).resolve().is_relative_to(args.source.resolve()):
        raise ValueError("PYTHONPATH must import ARDY from --source")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={args.source.resolve()}", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "693f74d13b3d04a0a22ce127ee79c929dd89756b":
        raise ValueError("Reference export requires the pinned ARDY source revision")
    from ardy.model import load_model
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    model = load_model("core", checkpoints_dir=str(args.checkpoints), text_encoder=False, device="cpu")
    path = args.checkpoints / "ARDY-Core-RP-20FPS-Horizon40"
    skeleton = model.skeleton
    joints = []
    for i, (name, parent) in enumerate(skeleton.bone_order_names_with_parents):
        p = None if parent is None else skeleton.bone_index[parent]
        offset = skeleton.neutral_joints[i].clone()
        if p is not None:
            offset -= skeleton.neutral_joints[p]
        joints.append(dict(name=name, parent=p, offset=offset.tolist(), bind_rotation=[0, 0, 0, 1]))
    stats = lambda folder: {kind: np.load(path / "stats" / folder / (kind+".npy")).astype(np.float32).tolist() for kind in ["mean", "std"]}
    config = dict(architecture="ardy-core-rp-v1", fps=20, horizon=40, frames_per_token=4,
                  motion_stats=stats("motion"), latent_stats=stats("post_quantization"),
                  skeleton=dict(id="ardy-core27", joints=joints))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    (args.out / "provenance.json").write_text(json.dumps(dict(source_revision=revision, checkpoint_revision="abe6c43beb28c867c950acb824b9c4ef3d63fb76", torch=torch.__version__), indent=2)+"\n")
    torch.manual_seed(8712)
    tensors = {}
    # A window with history, generation, sparse future, and all three CFG branches.
    b, h, t = 1, 8, 56
    x = torch.randn(b, t//4, 148)
    text = torch.randn(b, 1, 4096)*0.1
    heading = torch.tensor([0.4])
    obs = torch.zeros(b, t, 330)
    mask = torch.zeros_like(obs)
    for frame in [12, 39, 52]:
        mask[:, frame, [0, 2]] = 1
        obs[:, frame, [0, 2]] = torch.tensor([0.4, -0.2])
    idx = torch.arange(t)[None]
    hm = idx < h
    gm = (idx >= h) & (idx < h+40)
    fm = idx >= h+40
    htm, gtm, ftm = model.hybrid.convert_frame_mask_to_token_mask(hm, gm, fm, mask)
    kw = dict(history_len=torch.tensor([h]), generation_len=torch.tensor([40]), future_len=torch.tensor([t-h-40]), history_mask=hm, generation_mask=gm, future_mask=fm, history_token_mask=htm, generation_token_mask=gtm, future_token_mask=ftm, text_feat=text, text_feat_pad_mask=torch.ones(b,1,dtype=torch.bool), first_heading_angle=heading, motion_mask=mask, observed_motion=obs)
    with torch.inference_mode():
        pred = model.denoiser.model(x=x, timesteps=torch.tensor([7]), **kw)
        tensors.update(x=x, text=text, heading=heading[:,None], observed=obs, mask=mask, denoised=pred)
        sampled = x.clone()
        for step in range(9, -1, -1):
            sampled = model.denoising_step(sampled, kw['history_len'], kw['generation_len'], kw['future_len'], hm, gm, fm, htm, gtm, ftm, text, kw['text_feat_pad_mask'], torch.tensor([step]), heading, mask, obs, torch.tensor([10]), (2.0, 1.5))
        hybrid = sampled[:, :12]
        tensors['sampled'] = hybrid
        decoded = model.hybrid.get_explicit_motion_from_hybrid(hybrid, torch.ones(1,48,dtype=torch.bool),torch.tensor([48]))
        tensors['decoded'] = decoded
        tensors['encoded'] = model.hybrid.get_hybrid_motion_from_explicit(decoded,torch.tensor([48]),torch.ones(1,48,dtype=torch.bool))[0]
        tensors['local_root'] = model.motion_rep.global_root_to_local_root(x[:,:,:20].reshape(1,t,5), normalized=True, lengths=torch.tensor([48]))
        inv = model.motion_rep.inverse(decoded, is_normalized=True)
        tensors['joints'] = inv['posed_joints'].float()
        tensors['global_rotations'] = inv['global_rot_mats'].float()
    save_file({k:v.contiguous() for k,v in tensors.items()},args.out / "reference.safetensors")
    print(json.dumps({"source_revision":revision,"fixture_sha256":hashlib.sha256((args.out/'reference.safetensors').read_bytes()).hexdigest(),"tensors":{k:list(v.shape) for k,v in tensors.items()}},indent=2))


if __name__ == "__main__":
    main()
