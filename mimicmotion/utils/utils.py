import logging
from pathlib import Path

import numpy as np

try:
    from torchvision.io import write_video
except ImportError:
    write_video = None

logger = logging.getLogger(__name__)

def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if write_video is not None:
        try:
            write_video(save_path, frames, fps=fps)
            return
        except Exception:
            logger.warning("write_video failed (PyAV missing?), falling back to cv2")
    import cv2
    frames_np = frames.cpu().numpy() if hasattr(frames, 'numpy') else np.array(frames)
    h, w = frames_np.shape[1], frames_np.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

