from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class PoseNet(nn.Module):
    """a tiny conv network for introducing pose sequence as the condition
    """
    def __init__(self, noise_latent_channels=320, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # multiple convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1) * 2)

    def _initialize_weights(self):
        """Initialize weights with He. initialization and zero out the biases
        """
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, x):
        if x.ndim == 5:
            x = einops.rearrange(x, "b f c h w -> (b f) c h w")
        x = self.conv_layers(x)
        x = self.final_proj(x)

        return x * self.scale

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """load pretrained pose-net weights
        """
        if not Path(pretrained_model_path).exists():
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseNet's pretrained weights from {pretrained_model_path}.")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseNet(noise_latent_channels=320)

        model.load_state_dict(state_dict, strict=True)

        return model
