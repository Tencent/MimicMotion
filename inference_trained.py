"""
inference_trained.py

학습(train_base.py)에 실제로 사용된 원본 샘플에 대해, 학습된 체크포인트로
inference를 돌려서 overfit 학습이 그 샘플을 잘 재현하는지 확인하는 스크립트.
train_manifest에서 dataset.py와 동일한 전처리(리사이즈/크롭)로 프레임과 pose를
불러오기 때문에, video를 새로 dwpose로 추출하는 inference.py와 달리 학습 때
모델이 실제로 본 입력과 픽셀 단위로 동일하다.

실행:
    python inference_trained.py --config configs/inference_trained.yaml
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


import argparse
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dataset import MimicMotionFramesDataset, collate_fn
from inference import run_pipeline, device
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4


@torch.no_grad()
def main(args):
    cfg = OmegaConf.load(args.config)
    torch.set_default_dtype(torch.float16)

    ds = MimicMotionFramesDataset(
        str(cfg.train_manifest), int(cfg.resolution), int(cfg.num_frames),
    )
    sample_index = int(cfg.get("sample_index", 0))
    batch = collate_fn([ds[sample_index]])

    ref_image    = batch["ref_image"][0]    # (c, h, w), [-1, 1] -- same preprocessing as training
    ref_pose     = batch["ref_pose"][0]     # (c, h, w), [-1, 1]
    pose_images  = batch["pose_images"][0]  # (f, c, h, w), [-1, 1]

    # matches inference.preprocess(): image_pixels is the ref image only,
    # pose_pixels is [ref image's own pose, driving pose sequence...].
    # dataset.py casts to float32 explicitly, unlike preprocess()'s implicit
    # default-dtype promotion, so cast to fp16 here to match pose_net's dtype.
    image_pixels = ref_image.unsqueeze(0).to(torch.float16)
    pose_pixels = torch.cat([ref_pose.unsqueeze(0), pose_images], dim=0).to(torch.float16)

    infer_config = OmegaConf.create({
        "base_model_path": cfg.base_model_path,
        "ckpt_path": cfg.ckpt_path,
    })
    pipeline = create_pipeline(infer_config, device)

    task_config = OmegaConf.create({
        "seed": int(cfg.get("seed", 42)),
        "num_frames": int(cfg.get("tile_size", pose_pixels.size(0))),
        "frames_overlap": int(cfg.get("frames_overlap", 0)),
        "noise_aug_strength": float(cfg.get("noise_aug_strength", 0)),
        "num_inference_steps": int(cfg.get("num_inference_steps", 25)),
        "guidance_scale": float(cfg.get("guidance_scale", 2.0)),
    })

    video_frames = run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = Path(cfg.ckpt_path).stem
    out_path = output_dir / (
        f"trained_{ckpt_tag}_sample{sample_index}_"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    )
    save_to_mp4(video_frames, str(out_path), fps=int(cfg.get("fps", 15)))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference_trained.yaml")
    args = parser.parse_args()
    main(args)
