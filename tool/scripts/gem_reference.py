#!/usr/bin/env python3
"""Independent real-image GEM-X oracle: released ONNX + official Torch SAM/SOMA.

The public input contract is RGB. The upstream ViTPose crop helper expects BGR,
so this reference supplies BGR there and RGB to SAM, explicitly.
"""
import argparse,json,pathlib,sys,hashlib
import cv2
import numpy as np
import torch
import onnxruntime as ort
from export_gem_vision import pin_dinov3, SAM_CHECKPOINT
from model_export import verify_file

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ["upstream","soma-upstream","assets","soma-assets","image","out"]:ap.add_argument("--"+name,type=pathlib.Path,required=True)
    ap.add_argument("--crop",type=float,nargs=3,required=True);args=ap.parse_args()
    verify_file(args.assets/"sam3d_body.ckpt",SAM_CHECKPOINT)
    pin_dinov3()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path[:0]=[str(args.upstream),str(args.upstream/"third_party/sam-3d-body"),str(args.soma_upstream)]
    from gem.utils.vitpose_extractor import get_batch,keypoints_from_heatmaps,flip_heatmap_soma77
    from gem.utils.rotation_conversions import rotation_6d_to_matrix,matrix_to_axis_angle
    from gem.network.stats_compose import MM_V2_SOMA_METROSIM
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to
    from soma.body.soma import SOMALayer
    bgr=cv2.imread(str(args.image));rgb=bgr[:,:,::-1].copy();h,w=rgb.shape[:2]
    png=args.out.with_suffix(".png");cv2.imwrite(str(png),bgr)
    crop=torch.tensor([args.crop],dtype=torch.float32);cx,cy,size=args.crop
    focal=float(np.hypot(w,h));k=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],np.float32)
    inp,_=get_batch(bgr[None],crop);inp=inp[:,:,:,32:224].numpy()
    vit=ort.InferenceSession(str(args.assets/"onnx/vitpose.onnx"),providers=[("CUDAExecutionProvider",{"use_tf32":0}),"CPUExecutionProvider"])
    hm=vit.run(None,{"imgs":np.concatenate([inp,inp[:,:,:,::-1]],axis=0)})[0]
    hm=(hm[:1]+flip_heatmap_soma77(torch.from_numpy(hm[1:])).numpy())*.5
    kp,score=keypoints_from_heatmaps(hm,crop[:,:2].numpy(),np.array([[size*.75,size]],np.float32)/200)
    kp=np.concatenate([kp,score],axis=-1).astype(np.float32)
    del vit
    model,cfg=load_sam_3d_body(str(args.assets/"sam3d_body.ckpt"),device="cuda",mhr_path=str(args.soma_assets/"MHR/mhr_model_lod1.pt"));model.float().eval();model.backbone_dtype=torch.float32
    model.cfg.defrost();model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS=False;model.cfg.freeze()
    estimator=SAM3DBodyEstimator(sam_3d_body_model=model,model_cfg=cfg,human_detector=None,human_segmentor=None,fov_estimator=None)
    box=np.array([[cx-size/2,cy-size/2,cx+size/2,cy+size/2]],np.float32)
    batch=recursive_to(prepare_batch(rgb,estimator.transform,box,cam_int=torch.tensor(k[None])),"cuda")
    model._initialize_batch(batch);model.body_batch_idx=torch.tensor([0],device="cuda");model.hand_batch_idx=[]
    with torch.no_grad():
        inp_sam=model.data_preprocess(model._flatten_person(batch["img"]));emb=model.backbone(inp_sam)
        emb=emb+model._get_mask_prompt(batch,emb);batch["ray_cond"]=model.get_ray_condition(batch).flatten(0,1)
        token,_=model.forward_decoder(emb,condition_info=model._get_decoder_condition(batch),keypoints=torch.tensor([[[0.,0.,-2.]]],device="cuda"),batch=batch)
        token=token[:,0].cpu().numpy()
    sam_crop=batch["bbox_scale"].flatten().tolist()
    del model,estimator;torch.cuda.empty_cache()
    gem=ort.InferenceSession(str(args.assets/"onnx/gem_denoiser.onnx"),providers=[("CUDAExecutionProvider",{"use_tf32":0}),"CPUExecutionProvider"])
    feature,cam=gem.run(None,dict(obs=kp[None],bbx_xys=crop[None].numpy(),K_fullimg=k[None,None],f_imgseq=token[None],f_cam_angvel=np.array([[[1,0,0,0,1,0]]],np.float32)))
    stats=MM_V2_SOMA_METROSIM;decoded=torch.tensor(feature[0,0])*torch.tensor(stats["std"]).float().clamp_min(1)+torch.tensor(stats["mean"]).float()
    body=matrix_to_axis_angle(rotation_6d_to_matrix(decoded[:456].reshape(76,6)))
    root=matrix_to_axis_angle(rotation_6d_to_matrix(decoded[570:576]))
    rotations=torch.cat([root[None],body],dim=0)[None]
    identity=decoded[456:501][None];scales=decoded[502:570][None];global_scale=decoded[501].clamp(.7,1.).item()
    s,tx,ty=cam[0,0];sb=s*size+1e-9;translation=torch.tensor([[tx+2*(cx-w/2)/sb,ty+2*(cy-h/2)/sb,2*focal/sb]],dtype=torch.float32)
    soma=SOMALayer(data_root=args.soma_assets,device="cpu",identity_model_type="mhr",mode="torch",lod="mid");soma.skeleton_transfer.use_warp_for_rotations=False
    with torch.no_grad():
        soma.prepare_identity(identity,scales,repose_to_bind_pose=False,global_scale=global_scale)
        output=soma.pose(rotations,transl=translation,apply_correctives=False)
    result=dict(image=png.name,image_sha256=hashlib.file_digest(png.open("rb"),"sha256").hexdigest(),crop=dict(center=[cx,cy],size=size),camera=dict(focal=[focal,focal],center=[w/2,h/2]),sam_crop=sam_crop,keypoints=kp[0].tolist(),token=token[0].tolist(),features=feature[0,0].tolist(),pred_camera=cam[0,0].tolist(),identity=identity[0].tolist(),scales=scales[0].tolist(),global_scale=global_scale,rotations=rotations[0].tolist(),translation=translation[0].tolist(),vertices=output["vertices"][0].tolist(),joints=output["joints"][0].tolist())
    args.out.write_text(json.dumps(result)+"\n")
    # Small independent pixel fixtures catch affine/channel regressions without weights.
    np.save(args.out.with_name(args.out.stem+"-vit-input.npy"),inp)
    np.save(args.out.with_name(args.out.stem+"-sam-input.npy"),inp_sam.cpu().numpy())
    print(f"Saved real-image reference {args.out}",flush=True)

if __name__=="__main__":main()
