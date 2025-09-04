import math
from typing import Tuple

import torch
from torch import nn, Tensor


class ProprioTokenizer(nn.Module):
    """
    Encode flattened proprioceptive observations into a single Transformer token.

    Architecture:
      x [B, in_dim] -> MLP(2 layers, ReLU, optional dropout): in_dim->h1->h2 -> Linear: h2->token_dim
      Optional LayerNorm after the final projection.

    Default input layout (in_dim=45):
      | base_ang_vel (3) | projected_gravity (3) | velocity_commands (3) |
      | joint_pos (12)   | joint_vel (12)        | actions (12)          |
      | ... (extend if you add more terms) ...

    Args:
        in_dim: Input feature dimension.
        hidden_dims: Two hidden layer sizes for the MLP.
        token_dim: Output token dimension (must match vision token dim C).
        use_layernorm: If True, apply LayerNorm after projection.
        dropout: Dropout rate applied after each hidden ReLU (0.0 disables).
    """

    def __init__(
        self,
        in_dim: int = 45,
        hidden_dims: Tuple[int, int] = (256, 256),
        token_dim: int = 128,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError(f"hidden_dims must be a 2-tuple, got {hidden_dims}")

        h1, h2 = hidden_dims

        self.in_dim = in_dim
        self.token_dim = token_dim
        self.use_layernorm = use_layernorm

        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.proj = nn.Linear(h2, token_dim)

        self.act = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.drop2 = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.ln = nn.LayerNorm(token_dim) if use_layernorm else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming init for MLP (fan_in, ReLU)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        # Xavier for final projection
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, in_dim] proprioceptive features.

        Returns:
            t_prop: [B, token_dim] encoded token suitable to fuse with vision tokens.
        """
        if x.ndim != 2:
            raise AssertionError(f"Expected x with shape [B, {self.in_dim}], got rank {x.ndim} tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise AssertionError(f"Expected x.shape[1]=={self.in_dim}, got {x.shape[1]}")

        h = self.act(self.fc1(x))
        h = self.drop1(h)
        h = self.act(self.fc2(h))
        h = self.drop2(h)
        t_prop = self.proj(h)
        t_prop = self.ln(t_prop)
        return t_prop

if __name__ == "__main__":
    torch.manual_seed(0)
    encoder = ProprioTokenizer()  # defaults: in_dim=45, hidden_dims=(256,256), token_dim=128
    x = torch.randn(32, 45)  # [B, in_dim]
    out = encoder(x)
    print("Output shape:", tuple(out.shape))
    assert out.shape == (32, 128), f"Expected (32,128), got {tuple(out.shape)}"

    # Tiny fuse test
    # t_visual = torch.randn(32, 64, 128)  # 64 vision tokens per sample
    # fused = fuse_tokens(out, t_visual)
    # print("Fused shape:", tuple(fused.shape))
    # assert fused.shape == (32, 65, 128), f"Expected (32,65,128), got {tuple(fused.shape)}"