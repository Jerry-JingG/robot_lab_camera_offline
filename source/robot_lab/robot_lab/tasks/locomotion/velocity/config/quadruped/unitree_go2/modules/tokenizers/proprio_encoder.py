from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

__all__ = ["ProprioEncoder"]


class ProprioEncoder(nn.Module):
    """Encodes stacked proprioceptive histories into a single token."""

    def __init__(
        self,
        state_dim: int,
        hist_len: int = 3,
        token_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if hist_len <= 0:
            raise ValueError("hist_len must be positive.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")

        self.state_dim = state_dim
        self.hist_len = hist_len
        input_dim = state_dim * hist_len

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers.")
            block_layers = [
                self._build_linear(prev_dim, hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0.0:
                block_layers.append(nn.Dropout(dropout))
            if use_layernorm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Sequential(*block_layers))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.token_proj = self._build_linear(prev_dim, token_dim)
        self._use_residual = bool(layers) and input_dim == prev_dim

    def _build_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode stacked proprioception into a single token.

        Args:
            x: Tensor of shape [B, hist_len * state_dim].

        Returns:
            Tensor of shape [B, 1, token_dim].
        """
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D [batch, features].")
        expected_dim = self.state_dim * self.hist_len
        if x.size(1) != expected_dim:
            raise ValueError(
                f"Expected input feature dim {expected_dim}, got {x.size(1)}."
            )

        residual_source = x
        features = self.mlp(x)
        if self._use_residual:
            features = features + 0.1 * residual_source

        token = self.token_proj(features)
        return token.unsqueeze(1)

    @torch.no_grad()
    def infer_token(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation-mode inference helper."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(x)
        finally:
            if was_training:
                self.train()


if __name__ == "__main__":
    encoder = ProprioEncoder(state_dim=31, hist_len=3, token_dim=128)
    batch = torch.randn(4, 31 * 3)
    output = encoder(batch)
    assert output.shape == (4, 1, 128)
    assert torch.isfinite(output).all()
    print("ProprioEncoder test passed.")
