from typing import Optional

import torch
from torch import Tensor, nn


class DepthEncoder(nn.Module):
    """Encode stacked egocentric depth frames into visual tokens."""

    def __init__(
        self,
        in_frames: int = 4,
        in_size: int = 64,
        token_dim: int = 128,
        grid_size: int = 4,
        add_2d_pos_embed: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_frames <= 0:
            raise ValueError("in_frames must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")

        self.in_frames = in_frames
        self.in_size = in_size
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.add_2d_pos_embed = add_2d_pos_embed

        self.conv1 = nn.Conv2d(in_frames, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, token_dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.pool: Optional[nn.Module] = (
            None if grid_size == 4 else nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        self.drop = nn.Dropout(dropout)
        self.pos_embed: Optional[nn.Parameter]
        if add_2d_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(token_dim, grid_size, grid_size)
            )
        else:
            self.register_parameter("pos_embed", None)

        self._init_weights()

    @property
    def num_tokens(self) -> int:
        return self.grid_size * self.grid_size

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape [B, C, H, W], got {x.shape}.")
        if x.shape[1] != self.in_frames:
            raise ValueError(
                f"Expected {self.in_frames} input channels, got {x.shape[1]}."
            )
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            raise ValueError("Input height/width must be multiples of 2.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, in_frames, H, W].

        Returns:
            Tensor of shape [B, grid_size * grid_size, token_dim].
        """
        self._check_input(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.conv4(h)
        if self.pool is not None:
            h = self.pool(h)
        elif h.shape[-1] != self.grid_size or h.shape[-2] != self.grid_size:
            h = nn.functional.adaptive_avg_pool2d(h, (self.grid_size, self.grid_size))

        if self.pos_embed is not None:
            h = h + self.pos_embed.unsqueeze(0)

        tokens = h.flatten(2).transpose(1, 2).contiguous()
        tokens = self.drop(tokens)
        if not torch.isfinite(tokens).all():
            raise ValueError("Non-finite values detected in encoder output.")
        return tokens


if __name__ == "__main__":
    torch.manual_seed(0)

    encoder = DepthEncoder()
    batch = torch.randn(2, 4, 64, 64)
    out = encoder(batch)
    assert out.shape == (2, 16, 128), f"Unexpected shape {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values."

    encoder_small = DepthEncoder(grid_size=2)
    out_small = encoder_small(batch)
    assert out_small.shape == (2, 4, 128), f"Unexpected shape {out_small.shape}"
    assert torch.isfinite(out_small).all(), "Output contains non-finite values."
    print("DepthEncoder test passed.")