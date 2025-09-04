from __future__ import annotations

import torch
from torch import Tensor


def fuse_tokens(t_prop: Tensor, t_visual: Tensor) -> Tensor:
    """
    Fuse proprioceptive and visual tokens by simple concatenation along the token dimension.

    Args:
        t_prop: [B, D] proprio token.
        t_visual: [B, N, D] visual tokens (e.g., from a 4x4 grid -> N=16).

    Returns:
        tokens: [B, N+1, D], with proprio token placed first.
    """
    if t_prop.ndim != 2:
        raise AssertionError(f"t_prop must be [B, D], got shape {tuple(t_prop.shape)}")
    if t_visual.ndim != 3:
        raise AssertionError(f"t_visual must be [B, N, D], got shape {tuple(t_visual.shape)}")
    if t_prop.shape[0] != t_visual.shape[0]:
        raise AssertionError(
            f"Batch mismatch: t_prop B={t_prop.shape[0]} vs t_visual B={t_visual.shape[0]}"
        )
    if t_prop.shape[1] != t_visual.shape[2]:
        raise AssertionError(
            f"Dim mismatch: token_dim {t_prop.shape[1]} != {t_visual.shape[2]}"
        )

    # Concatenate along token axis, placing proprio first
    B, N, D = t_visual.shape
    return torch.cat([t_prop.unsqueeze(1), t_visual], dim=1).contiguous()
