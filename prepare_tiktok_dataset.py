"""
prepare_tiktok_dataset.py

/mnt/SSD/datasets/tiktok/TikTok_dataset/TikTok_dataset/000XX/{images,masks,densepose}
형식의 원본 TikTok Dataset(Jafarian & Park, CVPR 2021, 340 샘플)을 이 프로젝트의
train_vq.py / dataset.py가 요구하는 manifest 포맷으로 변환한다.

dataset.py::MimicMotionFramesDataset이 각 아이템에 요구하는 필드는 다음
4개뿐이다(실제로 읽어서 확인, 추측 아님 -- dataset.py의 __getitem__ 참고):
    frames_dir, pose_dir, ref_image, ref_pose
(sampled_indices/ref_index/source_video는 메타데이터일 뿐 로더가 사용하지
않는다.)

원본 TikTok Dataset에는 DWPose 스켈레톤이 없다(densepose UV map만 있음).
이 프로젝트의 PoseNet은 DWPose 스타일 스켈레톤 렌더링을 입력으로 기대하므로
(assets/example_data/videos/pose1*.mp4를 inference.py가 처리하는 방식과
동일), mimicmotion.dwpose.dwpose_detector로 직접 DWPose를 실행해 pose 이미지를
새로 생성한다.

mimicmotion/dwpose/preprocess.py::get_video_pose()는 decord.VideoReader(비디오
파일)만 받고 개별 PNG 프레임 리스트는 받지 못한다. 아래 get_frames_pose()는
get_video_pose()와 완전히 동일한 알고리즘(ref pose 검출 -> 같은 keypoint로
linear rescale 계수 계산 -> 각 프레임에 그 계수를 적용해 draw_pose)을 그대로
가져오되, 프레임 소스만 "video_path에서 decord로 읽기"에서 "이미 로드된 PNG
프레임 리스트"로 바꾼 것이다 -- 로직을 추측/변경하지 않았다.

출력 위치: /mnt/SSD/datasets/MimicMotion/train_dataset_tiktok/
    (기존 /mnt/SSD/datasets/MimicMotion/train_dataset/의 14개 샘플과 절대
    겹치지 않는 새 디렉토리 -- 기존 데이터는 전혀 건드리지 않는다.)

실행:
    /opt/conda/envs/mimicmotion/bin/python prepare_tiktok_dataset.py \
        --limit 5          # 먼저 5개 샘플만 시험
    /opt/conda/envs/mimicmotion/bin/python prepare_tiktok_dataset.py
                            # 전체 340개 샘플 처리
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from mimicmotion.dwpose.util import draw_pose
from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor

SOURCE_ROOT = Path("/mnt/SSD/datasets/tiktok/TikTok_dataset/TikTok_dataset")
OUTPUT_ROOT = Path("/mnt/SSD/datasets/MimicMotion/train_dataset_tiktok")
NUM_SAMPLED_FRAMES = 16
# matches this project's own pre-existing manifests (train_manifest.json /
# train_manifest_294.json both use ref_index=12 out of a 16-frame
# sampled_indices array) -- not a guess, an observed convention in this repo.
REF_INDEX = 12


def get_frames_pose(frames: List[np.ndarray], ref_image: np.ndarray) -> np.ndarray:
    """mimicmotion/dwpose/preprocess.py::get_video_pose()와 동일한 알고리즘.
    video_path 대신 이미 로드된 프레임 리스트를 직접 받는다는 점만 다르다."""
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [
        i for i in ref_keypoint_id
        if len(ref_pose["bodies"]["subset"]) > 0 and ref_pose["bodies"]["subset"][0][i] >= 0.0
    ]
    if not ref_keypoint_id:
        raise RuntimeError("no valid keypoints detected on reference frame")
    ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id]

    height, width, _ = ref_image.shape

    detected_poses = [dwprocessor(frm) for frm in frames]

    detected_bodies = np.stack([
        p["bodies"]["candidate"] for p in detected_poses if p["bodies"]["candidate"].shape[0] == 18
    ])
    if detected_bodies.shape[0] == 0:
        raise RuntimeError("no frame had a full 18-keypoint detection; cannot fit rescale")
    detected_bodies = detected_bodies[:, ref_keypoint_id]

    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1
    )
    fh, fw, _ = frames[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])

    output_pose = []
    for detected_pose in detected_poses:
        detected_pose["bodies"]["candidate"] = detected_pose["bodies"]["candidate"] * a + b
        detected_pose["faces"] = detected_pose["faces"] * a + b
        detected_pose["hands"] = detected_pose["hands"] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def get_image_pose(ref_image: np.ndarray) -> np.ndarray:
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    return np.array(draw_pose(ref_pose, height, width))


def _sorted_frame_files(images_dir: Path) -> List[Path]:
    return sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))


def process_sample(tiktok_id: str, out_dir: Path) -> Optional[str]:
    """Returns an error message string on failure, None on success."""
    images_dir = SOURCE_ROOT / tiktok_id / "images"
    if not images_dir.is_dir():
        return f"no images/ dir"

    frame_files = _sorted_frame_files(images_dir)
    if len(frame_files) < 8:
        return f"only {len(frame_files)} frames (<8, need at least num_frames=8)"

    if len(frame_files) >= NUM_SAMPLED_FRAMES:
        idx = np.linspace(0, len(frame_files) - 1, NUM_SAMPLED_FRAMES).round().astype(int)
        idx = sorted(set(idx.tolist()))
    else:
        idx = list(range(len(frame_files)))
    sampled_files = [frame_files[i] for i in idx]

    frames_rgb = [np.array(Image.open(p).convert("RGB")) for p in sampled_files]
    ref_i = REF_INDEX if REF_INDEX < len(frames_rgb) else len(frames_rgb) // 2
    ref_image = frames_rgb[ref_i]

    try:
        pose_frames = get_frames_pose(frames_rgb, ref_image)
        ref_pose_img = get_image_pose(ref_image)
    except Exception as e:  # noqa: BLE001
        return f"DWPose failed: {e}"
    finally:
        dwprocessor.release_memory()

    frames_out = out_dir / "frames"
    poses_out = out_dir / "poses"
    frames_out.mkdir(parents=True, exist_ok=True)
    poses_out.mkdir(parents=True, exist_ok=True)

    # draw_pose() (mimicmotion/dwpose/util.py) returns CHW (it ends with
    # `.transpose(2, 0, 1)`), matching what inference.py expects to stack
    # into a (T, C, H, W) tensor directly -- but PIL.Image.fromarray needs
    # HWC, so transpose back only for saving as PNG here.
    for i, frm in enumerate(frames_rgb):
        Image.fromarray(frm).save(frames_out / f"{i:04d}.png")
    for i, pose in enumerate(pose_frames):
        Image.fromarray(pose.transpose(1, 2, 0)).save(poses_out / f"{i:04d}.png")
    Image.fromarray(ref_image).save(out_dir / "ref_image.png")
    Image.fromarray(ref_pose_img.transpose(1, 2, 0)).save(out_dir / "ref_pose.png")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tiktok samples")
    parser.add_argument("--start", type=int, default=0, help="skip the first N tiktok samples (for resuming)")
    args = parser.parse_args()

    tiktok_ids = sorted(p.name for p in SOURCE_ROOT.iterdir() if p.is_dir())
    if args.start:
        tiktok_ids = tiktok_ids[args.start:]
    if args.limit:
        tiktok_ids = tiktok_ids[: args.limit]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    failures = []

    for n, tiktok_id in enumerate(tiktok_ids):
        out_name = f"sample_tiktok_{tiktok_id}"
        out_dir = OUTPUT_ROOT / out_name
        try:
            err = process_sample(tiktok_id, out_dir)
        except Exception:  # noqa: BLE001
            err = f"unexpected exception:\n{traceback.format_exc()}"

        if err is not None:
            failures.append((tiktok_id, err))
            print(f"[{n+1}/{len(tiktok_ids)}] SKIP {tiktok_id}: {err}")
            continue

        manifest_items.append({
            "frames_dir": str(out_dir / "frames"),
            "pose_dir": str(out_dir / "poses"),
            "ref_image": str(out_dir / "ref_image.png"),
            "ref_pose": str(out_dir / "ref_pose.png"),
            "source_dataset": "tiktok",
            "source_tiktok_id": tiktok_id,
        })
        print(f"[{n+1}/{len(tiktok_ids)}] OK {tiktok_id} -> {out_name}")

    manifest_path = OUTPUT_ROOT / "train_manifest_tiktok.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_items, f, indent=2)

    print(f"\nDone: {len(manifest_items)} succeeded, {len(failures)} failed out of {len(tiktok_ids)}")
    print(f"Manifest written to: {manifest_path}")
    if failures:
        print("Failures:")
        for tid, err in failures:
            print(f"  {tid}: {err}")


if __name__ == "__main__":
    main()
