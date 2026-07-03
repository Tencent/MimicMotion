from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, resize

ASPECT_RATIO = 9.0 / 16.0


def _target_hw(h: int, w: int, resolution: int) -> Tuple[int, int]:
    if h > w:
        w_t = resolution
        h_t = int(resolution / ASPECT_RATIO // 64) * 64
    else:
        h_t = resolution
        w_t = int(resolution / ASPECT_RATIO // 64) * 64
    return h_t, w_t


def _resize_crop(img: torch.Tensor, resolution: int) -> torch.Tensor:
    h, w = img.shape[-2:]
    h_t, w_t = _target_hw(h, w, resolution)
    ratio = float(h) / float(w)
    if ratio < h_t / w_t:
        h_r, w_r = h_t, math.ceil(h_t / ratio)
    else:
        h_r, w_r = math.ceil(w_t * ratio), w_t
    img = resize(img, [h_r, w_r], antialias=None)
    return center_crop(img, [h_t, w_t])


def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 127.5 - 1.0


class MimicMotionFramesDataset(Dataset):
    def __init__(self, manifest_path: str, resolution: int, num_frames: int):
        self.items: List[Dict[str, Any]] = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        self.resolution = resolution
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.items)

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = read_image(path)[:3]
        h_t, w_t = _target_hw(img.shape[-2], img.shape[-1], self.resolution)
        if img.shape[-2:] != (h_t, w_t):
            img = _resize_crop(img, self.resolution)
        return img

    def _load_sequence(self, folder: str) -> torch.Tensor:
        files = sorted(Path(folder).glob("*.png"))
        if len(files) < self.num_frames:
            raise ValueError(
                f"{folder} has {len(files)} png files, need {self.num_frames}"
            )
        return torch.stack([self._load_rgb(str(p)) for p in files[: self.num_frames]])

    def _load_masks(
        self, folder: Optional[str], out_hw: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        if not folder:
            return None
        h, w = out_hw
        masks = []
        for p in sorted(Path(folder).glob("*.png"))[: self.num_frames]:
            m = read_image(str(p))[:1].float() / 255.0
            if m.shape[-2:] != (h, w):
                m = resize(m, [h, w], antialias=None)
            masks.append(m[0])
        return torch.stack(masks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        frames = self._load_sequence(item["frames_dir"])
        poses  = self._load_sequence(item["pose_dir"])
        ref    = self._load_rgb(item["ref_image"])
        ref_pose = self._load_rgb(item["ref_pose"]) if item.get("ref_pose") else poses[0]
        masks  = self._load_masks(item.get("hand_mask_dir"), tuple(frames.shape[-2:]))
        return {
            "pixel_values": _to_float(frames),
            "pose_images":  _to_float(poses),
            "ref_image":    _to_float(ref),
            "ref_pose":     _to_float(ref_pose),
            "hand_masks":   masks,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    masks = None
    if batch[0]["hand_masks"] is not None:
        masks = torch.stack([b["hand_masks"] for b in batch])
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "pose_images":  torch.stack([b["pose_images"]  for b in batch]),
        "ref_image":    torch.stack([b["ref_image"]    for b in batch]),
        "ref_pose":     torch.stack([b["ref_pose"]     for b in batch]),
        "hand_masks":   masks,
    }
