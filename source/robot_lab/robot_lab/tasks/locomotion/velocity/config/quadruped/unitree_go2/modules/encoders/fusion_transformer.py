from typing import Dict, Optional

import torch
from torch import Tensor, nn

__all__ = ["MultiModalFusionTransformer"]


class MultiHeadSelfAttention(nn.Module):
    """Minimal multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores + self._broadcast_mask(attn_mask, attn_scores)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().reshape(bsz, seq_len, dim)
        out = self.proj(context)
        out = self.proj_drop(out)
        return out

    @staticmethod
    def _broadcast_mask(mask: Tensor, target: Tensor) -> Tensor:
        """Convert bool/float masks to additive masks broadcastable to target."""
        mask = mask.to(device=target.device)
        if mask.dtype == torch.bool:
            zero = torch.zeros(1, dtype=target.dtype, device=target.device)
            neg_inf = torch.full((1,), float("-inf"), dtype=target.dtype, device=target.device)
            mask = torch.where(mask, zero, neg_inf)
        else:
            mask = mask.to(dtype=target.dtype)
        return mask


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with optional pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        norm_first: bool,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout, dropout)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        if self.norm_first:
            x = x + self.attn(self.norm1(x), attn_mask)
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out = self.attn(x, attn_mask)
            x = self.norm1(x + attn_out)
            ffn_out = self.mlp(x)
            x = self.norm2(x + ffn_out)
        return x


class MultiModalFusionTransformer(nn.Module):
    """Fuses proprioceptive and visual tokens with a lightweight Transformer."""

    def __init__(
        self,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        add_modality_embed: bool = True,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError("num_heads must divide token_dim.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")

        self.token_dim = token_dim
        self.add_modality_embed = add_modality_embed

        if add_modality_embed:
            self.prop_embed = nn.Parameter(torch.zeros(1, 1, token_dim))
            self.vis_embed = nn.Parameter(torch.zeros(1, 1, token_dim))
        else:
            self.register_parameter("prop_embed", None)
            self.register_parameter("vis_embed", None)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(token_dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_inputs(self, prop: Tensor, vis: Tensor) -> None:
        if prop.ndim != 3 or vis.ndim != 3:
            raise ValueError("prop and vis must be 3D tensors [B, T, C].")
        if prop.shape[1] != 1:
            raise ValueError("prop must have exactly one token.")
        if prop.shape[0] != vis.shape[0]:
            raise ValueError("prop and vis must share the batch dimension.")
        if prop.shape[2] != self.token_dim or vis.shape[2] != self.token_dim:
            raise ValueError("Token dimensions must match token_dim.")
        if vis.shape[1] == 0:
            raise ValueError("vis must contain at least one token.")

    def forward(
        self,
        prop: Tensor,
        vis: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            prop: Tensor of shape [B, 1, C].
            vis: Tensor of shape [B, T, C].
            attn_mask: Optional mask broadcastable to [B, num_heads, 1 + T, 1 + T].

        Returns:
            Dict with fused sequence and pooled modality features.
        """
        self._validate_inputs(prop, vis)
        if self.add_modality_embed:
            prop = prop + self.prop_embed
            vis = vis + self.vis_embed
        x = torch.cat([prop, vis], dim=1)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)

        prop_tokens = x[:, :1, :]
        vis_tokens = x[:, 1:, :]
        prop_pooled = prop_tokens.mean(dim=1)
        vis_pooled = vis_tokens.mean(dim=1)
        all_pooled = x.mean(dim=1)

        return {
            "seq": x,
            "prop_pooled": prop_pooled,
            "vis_pooled": vis_pooled,
            "all_pooled": all_pooled,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    try:
        from modules.tokenizers.proprio_encoder import ProprioEncoder  # type: ignore
        from modules.tokenizers.depth_encoder import DepthEncoder  # type: ignore
    except ModuleNotFoundError:
        from proprio_encoder import ProprioEncoder  # type: ignore
        from depth_encoder import DepthEncoder  # type: ignore

    proprio_encoder = ProprioEncoder(state_dim=31, hist_len=3, token_dim=128)
    depth_encoder = DepthEncoder(in_frames=4, in_size=64, token_dim=128, grid_size=4)
    prop_input = torch.randn(2, 31 * 3)
    depth_input = torch.randn(2, 4, 64, 64)

    prop_tokens = proprio_encoder(prop_input)
    vis_tokens = depth_encoder(depth_input)
    assert prop_tokens.shape == (2, 1, 128)
    assert vis_tokens.shape == (2, 16, 128)

    fusion = MultiModalFusionTransformer()
    fused = fusion(prop_tokens, vis_tokens)

    assert fused["seq"].shape == (2, 17, 128)
    for key in ("prop_pooled", "vis_pooled", "all_pooled"):
        tensor = fused[key]
        assert tensor.shape == (2, 128)
        assert torch.isfinite(tensor).all()

    print("MultiModalFusionTransformer smoke test passed.")
