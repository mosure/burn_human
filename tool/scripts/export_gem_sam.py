#!/usr/bin/env python3
"""Export SAM3D's legacy GEM-X body-token decoder, with an independent Torch oracle."""
import argparse
import hashlib
import json
import pathlib
import sys
import torch
import numpy as np
from model_export import ExportWriter,verify_file
from export_gem_vision import REVISION,SOURCE,SAM_CHECKPOINT,pin_dinov3

BODY_TRIPLES=[(0,2,4),(6,8,10),(12,13,14),(15,16,17),(18,19,20),(21,22,23),(24,25,26),(27,28,29),(34,35,36),(37,38,39),(44,45,46),(53,54,55),(64,65,66),(85,69,73),(86,70,79),(87,71,82),(88,72,76),(91,92,93),(112,96,100),(113,97,106),(114,98,109),(115,99,103),(130,131,132)]
BODY_SINGLE=[1,3,5,7,9,11,30,31,32,33,40,41,42,43,47,48,49,50,51,52,56,57,58,59,60,61,62,63,67,68,74,75,77,78,80,81,83,84,89,90,94,95,101,102,104,105,107,108,110,111,116,117,118,119,120,121,122,123]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets",type=pathlib.Path,required=True)
    ap.add_argument("--upstream",type=pathlib.Path,required=True)
    ap.add_argument("--mhr",type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,required=True)
    ap.add_argument("--vision-reference",type=pathlib.Path,required=True)
    args=ap.parse_args()
    verify_file(args.assets/"sam3d_body.ckpt",SAM_CHECKPOINT)
    pin_dinov3()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(args.upstream/"third_party/sam-3d-body"))
    from sam_3d_body import load_sam_3d_body
    model,cfg=load_sam_3d_body(str(args.assets/"sam3d_body.ckpt"),device="cuda",mhr_path=str(args.mhr))
    model=model.float().eval();model.cfg.defrost();model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS=False;model.cfg.freeze()
    writer=ExportWriter(args.out)
    prefixes=["head_pose.proj.","head_camera.proj.","init_pose.","init_camera.","init_to_token_mhr.","prev_to_token_mhr.","prompt_to_token.","keypoint_embedding.","keypoint3d_embedding.","keypoint_posemb_linear.","keypoint3d_posemb_linear.","keypoint_feat_linear.","ray_cond_emb.","decoder."]
    for k,v in model.state_dict().items():
        if any(k.startswith(p) for p in prefixes):
            if k=="ray_cond_emb.conv.weight":v=v[:,:,0,0]
            writer.tensor(k,v)
    with torch.no_grad():
        point=torch.tensor([[[0.,0.,-2.]]],device="cuda")
        embedding,_=model.prompt_encoder(keypoints=point)
        writer.tensor("prompt.invalid",embedding)
        writer.tensor("prompt.image_pe",model.prompt_encoder.get_dense_pe((32,32)).flatten(2).transpose(1,2))
        mask,no_mask=model.prompt_encoder.get_mask_embeddings(torch.zeros((1,1,512,512),device="cuda"),1,(32,32))
        writer.tensor("prompt.no_mask",no_mask[0,:,0,0])
    head=model.head_pose
    assert torch.equal(head.hand_pose_comps,torch.eye(54,device="cuda"))
    writer.tensor("head_pose.keypoint_mapping",head.keypoint_mapping[:70])
    meta=dict(scale_mean=head.scale_mean.tolist(),scale_comps=head.scale_comps.tolist(),hand_mean=head.hand_pose_mean.tolist(),hand_left=head.hand_joint_idxs_left.tolist(),hand_right=head.hand_joint_idxs_right.tolist(),body_triples=BODY_TRIPLES,body_single=BODY_SINGLE)
    writer.json("head.json",meta)
    writer.asset("LICENSE",(args.upstream/"third_party/sam-3d-body/LICENSE").read_bytes())
    writer.json("provenance.json",dict(revision=REVISION,source_revision=SOURCE,sam_source_revision="b5c765a0d89d789985e186d396315e7590887b94",checkpoint_sha256=hashlib.file_digest((args.assets/"sam3d_body.ckpt").open("rb"),"sha256").hexdigest(),contract="legacy single primary body token; no mask or hand detector; six iterative MHR predictions"))
    writer.finish(model="nvidia/GEM-X:sam-decoder",model_revision=REVISION,source_revision=SOURCE,converter="export_gem_sam.py:v1",license="SAM License; see metadata/LICENSE",config=dict(layers=6,hidden=1024,heads=8,head_dim=64,image_height=32,image_width=32,image_channels=1280,pose_dimensions=519,keypoints=70))
    ref=json.loads(args.vision_reference.read_text())
    image=torch.tensor(ref["output"],device="cuda").reshape(ref["output_shape"])
    batch=dict(img=torch.zeros((1,1,3,512,512),device="cuda"),bbox_center=torch.tensor([[[320.,240.]]],device="cuda"),bbox_scale=torch.tensor([[[400.,400.]]],device="cuda"),ori_img_size=torch.tensor([[[640.,480.]]],device="cuda"),img_size=torch.tensor([[[512.,512.]]],device="cuda"),cam_int=torch.tensor([[[800.,0.,320.],[0.,800.,240.],[0.,0.,1.]]],device="cuda"),affine_trans=torch.tensor([[[[1.28,0.,-153.6],[0.,1.28,-51.2]]]],device="cuda"))
    model._max_num_person=1;model._batch_size=1
    model.body_batch_idx=torch.tensor([0],device="cuda");model.hand_batch_idx=[]
    batch["ray_cond"]=model.get_ray_condition(batch).flatten(0,1)
    cond=model._get_decoder_condition(batch)
    with torch.no_grad():
        tokens,poses=model.forward_decoder(image+no_mask,condition_info=cond,keypoints=point,batch=batch)
        ray_emb=model.ray_cond_emb(image+no_mask,batch["ray_cond"])
    reference=dict(revision=REVISION,embedding=image.flatten().tolist(),crop=dict(center=[320.,240.],size=400.),camera=dict(focal=[800.,800.],center=[320.,240.]),token=tokens[0,0].tolist(),ray_embedding=ray_emb.flatten().tolist(),layers=[dict(params=p["mhr_model_params"][0].tolist(),shape=p["shape"][0].tolist(),keypoints=p["pred_keypoints_3d"][0].tolist(),camera=p["pred_cam"][0].tolist()) for p in poses])
    (args.out/"reference.json").write_text(json.dumps(reference)+"\n")
    print(f"Exported SAM decoder {len(writer.tensors)} tensors",flush=True)

if __name__=="__main__":main()
