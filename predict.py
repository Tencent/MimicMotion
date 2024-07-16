# predict.py
import subprocess
import time
from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from datetime import datetime

from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from constants import ASPECT_RATIO

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

BASE_URL = f"https://weights.replicate.delivery/default/MimicMotion/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [
            "DWPose.tar",
            "MimicMotion.pth",
            "MimicMotion_1-1.pth",
            "SVD.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Move imports here and make them global
        # This ensures model files are downloaded before importing mimicmotion modules
        global MimicMotionPipeline, create_pipeline, save_to_mp4, get_video_pose, get_image_pose
        from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
        from mimicmotion.utils.loader import create_pipeline
        from mimicmotion.utils.utils import save_to_mp4
        from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

        # Load config with new checkpoint as default
        self.config = OmegaConf.create(
            {
                "base_model_path": "models/SVD/stable-video-diffusion-img2vid-xt-1-1",
                "ckpt_path": "models/MimicMotion_1-1.pth",
            }
        )

        # Create the pipeline with the new checkpoint
        self.pipeline = create_pipeline(self.config, self.device)
        self.current_checkpoint = "v1-1"
        self.current_dtype = torch.get_default_dtype()

    def predict(
        self,
        motion_video: Path = Input(
            description="Reference video file containing the motion to be mimicked"
        ),
        appearance_image: Path = Input(
            description="Reference image file for the appearance of the generated video"
        ),
        resolution: int = Input(
            description="Height of the output video in pixels. Width is automatically calculated.",
            default=576,
            ge=64,
            le=1024,
        ),
        chunk_size: int = Input(
            description="Number of frames to generate in each processing chunk",
            default=16,
            ge=2,
        ),
        frames_overlap: int = Input(
            description="Number of overlapping frames between chunks for smoother transitions",
            default=6,
            ge=0,
        ),
        denoising_steps: int = Input(
            description="Number of denoising steps in the diffusion process. More steps can improve quality but increase processing time.",
            default=25,
            ge=1,
            le=100,
        ),
        noise_strength: float = Input(
            description="Strength of noise augmentation. Higher values add more variation but may reduce coherence with the reference.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        guidance_scale: float = Input(
            description="Strength of guidance towards the reference. Higher values adhere more closely to the reference but may reduce creativity.",
            default=2.0,
            ge=0.1,
            le=10.0,
        ),
        sample_stride: int = Input(
            description="Interval for sampling frames from the reference video. Higher values skip more frames.",
            default=2,
            ge=1,
        ),
        output_frames_per_second: int = Input(
            description="Frames per second of the output video. Affects playback speed.",
            default=15,
            ge=1,
            le=60,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
        checkpoint_version: str = Input(
            description="Choose the checkpoint version to use",
            choices=["v1", "v1-1"],
            default="v1-1",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        ref_video = motion_video
        ref_image = appearance_image
        num_frames = chunk_size
        num_inference_steps = denoising_steps
        noise_aug_strength = noise_strength
        fps = output_frames_per_second
        use_fp16 = True

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        need_pipeline_update = False

        # Check if we need to switch checkpoints
        if checkpoint_version != self.current_checkpoint:
            if checkpoint_version == "v1":
                self.config.ckpt_path = "models/MimicMotion.pth"
            else:  # v1-1
                self.config.ckpt_path = "models/MimicMotion_1-1.pth"
            need_pipeline_update = True
            self.current_checkpoint = checkpoint_version

        # Check if we need to switch dtype
        target_dtype = torch.float16 if use_fp16 else torch.float32
        if target_dtype != self.current_dtype:
            torch.set_default_dtype(target_dtype)
            need_pipeline_update = True
            self.current_dtype = target_dtype

        # Update pipeline if needed
        if need_pipeline_update:
            print(
                f"Updating pipeline with checkpoint: {self.config.ckpt_path} and dtype: {torch.get_default_dtype()}"
            )
            self.pipeline = create_pipeline(self.config, self.device)

        print(f"Using checkpoint: {self.config.ckpt_path}")
        print(f"Using dtype: {torch.get_default_dtype()}")

        print(
            f"[!] ({type(ref_video)}) ref_video={ref_video}, "
            f"[!] ({type(ref_image)}) ref_image={ref_image}, "
            f"[!] ({type(resolution)}) resolution={resolution}, "
            f"[!] ({type(num_frames)}) num_frames={num_frames}, "
            f"[!] ({type(frames_overlap)}) frames_overlap={frames_overlap}, "
            f"[!] ({type(num_inference_steps)}) num_inference_steps={num_inference_steps}, "
            f"[!] ({type(noise_aug_strength)}) noise_aug_strength={noise_aug_strength}, "
            f"[!] ({type(guidance_scale)}) guidance_scale={guidance_scale}, "
            f"[!] ({type(sample_stride)}) sample_stride={sample_stride}, "
            f"[!] ({type(fps)}) fps={fps}, "
            f"[!] ({type(seed)}) seed={seed}, "
            f"[!] ({type(use_fp16)}) use_fp16={use_fp16}"
        )

        # Input validation
        if not ref_video.exists():
            raise ValueError(f"Reference video file does not exist: {ref_video}")
        if not ref_image.exists():
            raise ValueError(f"Reference image file does not exist: {ref_image}")

        if resolution % 8 != 0:
            raise ValueError(f"Resolution must be a multiple of 8, got {resolution}")

        if resolution < 64 or resolution > 1024:
            raise ValueError(
                f"Resolution must be between 64 and 1024, got {resolution}"
            )

        if num_frames <= frames_overlap:
            raise ValueError(
                f"Number of frames ({num_frames}) must be greater than frames overlap ({frames_overlap})"
            )

        if num_frames < 2:
            raise ValueError(f"Number of frames must be at least 2, got {num_frames}")

        if frames_overlap < 0:
            raise ValueError(
                f"Frames overlap must be non-negative, got {frames_overlap}"
            )

        if num_inference_steps < 1 or num_inference_steps > 100:
            raise ValueError(
                f"Number of inference steps must be between 1 and 100, got {num_inference_steps}"
            )

        if noise_aug_strength < 0.0 or noise_aug_strength > 1.0:
            raise ValueError(
                f"Noise augmentation strength must be between 0.0 and 1.0, got {noise_aug_strength}"
            )

        if guidance_scale < 0.1 or guidance_scale > 10.0:
            raise ValueError(
                f"Guidance scale must be between 0.1 and 10.0, got {guidance_scale}"
            )

        if sample_stride < 1:
            raise ValueError(f"Sample stride must be at least 1, got {sample_stride}")

        if fps < 1 or fps > 60:
            raise ValueError(f"FPS must be between 1 and 60, got {fps}")

        try:
            # Preprocess
            pose_pixels, image_pixels = self.preprocess(
                str(ref_video),
                str(ref_image),
                resolution=resolution,
                sample_stride=sample_stride,
            )

            # Run pipeline
            video_frames = self.run_pipeline(
                image_pixels,
                pose_pixels,
                num_frames=num_frames,
                frames_overlap=frames_overlap,
                num_inference_steps=num_inference_steps,
                noise_aug_strength=noise_aug_strength,
                guidance_scale=guidance_scale,
                seed=seed,
            )

            # Save output
            output_path = f"/tmp/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            save_to_mp4(video_frames, output_path, fps=fps)

            return Path(output_path)

        except Exception as e:
            print(f"An error occurred during prediction: {str(e)}")
            raise

    def preprocess(self, video_path, image_path, resolution=576, sample_stride=2):
        image_pixels = Image.open(image_path).convert("RGB")
        image_pixels = pil_to_tensor(image_pixels)  # (c, h, w)
        h, w = image_pixels.shape[-2:]

        if h > w:
            w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
        else:
            w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution

        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, int(h_target / h_w_ratio)
        else:
            h_resize, w_resize = int(w_target * h_w_ratio), w_target

        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()

        image_pose = get_image_pose(image_pixels)
        video_pose = get_video_pose(
            video_path, image_pixels, sample_stride=sample_stride
        )

        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

        return (
            torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
            torch.from_numpy(image_pixels) / 127.5 - 1,
        )

    def run_pipeline(
        self,
        image_pixels,
        pose_pixels,
        num_frames,
        frames_overlap,
        num_inference_steps,
        noise_aug_strength,
        guidance_scale,
        seed,
    ):
        image_pixels = [
            Image.fromarray(
                (img.cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
            )
            for img in image_pixels
        ]
        pose_pixels = pose_pixels.unsqueeze(0).to(self.device)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        frames = self.pipeline(
            image_pixels,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(1),
            tile_size=num_frames,
            tile_overlap=frames_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=7,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
            decode_chunk_size=8,
            output_type="pt",
            device=self.device,
        ).frames.cpu()

        video_frames = (frames * 255.0).to(torch.uint8)
        return video_frames[0, 1:]  # Remove the first frame (reference image)
