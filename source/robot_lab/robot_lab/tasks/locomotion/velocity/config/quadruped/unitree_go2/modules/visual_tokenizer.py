from typing import Tuple

import torch
from torch import nn, Tensor


class VisualTokenizer(nn.Module):
    """
    Tokenize stacked depth images into a sequence of visual tokens for cross-modal Transformers.

    Input:
        x: [batch_size, 4, 64, 64]  (4 stacked grayscale depth frames)

    Encoder (DQN-style, Mnih et al. 2015):
        - Conv2d(4 -> 32, kernel=8, stride=4) + ReLU  -> [B, 32, 15, 15]
        - Conv2d(32 -> 64, kernel=4, stride=2) + ReLU -> [B, 64,  6,  6]
        - Conv2d(64 -> 128, kernel=3, stride=1) + ReLU-> [B, 128, 4,  4]

    Tokenization:
        Flatten spatial dims to tokens and transpose to [batch_size, 16, 128]
        (16 = 4 * 4 tokens, each 128-dim).

    No additional projection; feature vectors are used directly as tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 4, 64, 64] stacked depth images

        Returns:
            tokens: [B, 16, 128] (flattened 4x4 feature map; 16 tokens, 128-dim each)
        """
        if x.ndim != 4:
            raise AssertionError(f"Expected input rank 4 [B,4,64,64], got rank {x.ndim} with shape {tuple(x.shape)}")
        if x.shape[1:] != (4, 64, 64):
            raise AssertionError(f"Expected input shape [B,4,64,64], got {tuple(x.shape)}")

        feat = self.encoder(x)  # [B, 128, 4, 4]
        if feat.shape[1:] != (128, 4, 4):
            raise AssertionError(
                f"Encoder produced unexpected shape {tuple(feat.shape)}; expected [B,128,4,4]."
            )
        # Flatten spatial dims to tokens, then transpose to [B, num_tokens, dim]
        B, C, H, W = feat.shape
        tokens = feat.view(B, C, H * W).transpose(1, 2).contiguous()  # [B, 16, 128]
        return tokens
