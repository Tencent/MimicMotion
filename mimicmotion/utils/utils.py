import logging
from pathlib import Path

from torchvision.io import write_video

logger = logging.getLogger(__name__)

def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)

