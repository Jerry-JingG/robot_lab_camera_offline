"""
Transformer-XL (TXL) backbone for PPO that fuses proprioceptive and visual tokens,
implements relative-position multi-head attention with memory, stacks TXL blocks,
and provides simple modality-wise pooling heads for policy and value.

No external dependencies beyond torch and typing. Python 3.10+.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
from torch import Tensor, nn


# ------------------------------
# Token utilities
# ------------------------------

def fuse_tokens(t_prop: Tensor, t_vis: Tensor, prop_first: bool = True) -> Tensor:
    """Fuse proprioceptive and visual tokens.

    Args:
        t_prop: [B, C] single proprio token per batch item.
        t_vis:  [B, N, C] visual tokens per batch item.
        prop_first: If True, place proprio token at index 0, else append at end.

    Returns:
        tokens: [B, 1+N, C].
    """
    if t_prop.ndim != 2:
        raise ValueError(f"t_prop must be [B, C], got shape {tuple(t_prop.shape)}")
    if t_vis.ndim != 3:
        raise ValueError(f"t_vis must be [B, N, C], got shape {tuple(t_vis.shape)}")
    if t_prop.shape[0] != t_vis.shape[0]:
        raise ValueError(
            f"Batch mismatch: t_prop B={t_prop.shape[0]} vs t_vis B={t_vis.shape[0]}"
        )
    if t_prop.shape[1] != t_vis.shape[2]:
        raise ValueError(
            f"Channel mismatch: C_prop={t_prop.shape[1]} vs C_vis={t_vis.shape[2]}"
        )
    B, N, C = t_vis.shape
    if prop_first:
        return torch.cat([t_prop.unsqueeze(1), t_vis], dim=1)
    else:
        return torch.cat([t_vis, t_prop.unsqueeze(1)], dim=1)


# ------------------------------
# Masks and relative positions
# ------------------------------

def build_attn_mask_from_token_masks(
    mask_now: Optional[Tensor],
    mask_mem: Optional[Tensor],
) -> Optional[Tensor]:
    """Build a 3D attention mask from per-token masks.

    Args:
        mask_now: [B, T_now] boolean; True indicates the token is invalid (to be masked)
                  for queries. If None, treated as all False.
        mask_mem: [B, T_mem] boolean; True indicates invalid memory tokens (to be masked)
                  for keys. If None, treated as all False.

    Returns:
        attn_mask: [B, T_now, T_mem+T_now] boolean. True => masked (not allowed).
    """
    if mask_now is None and mask_mem is None:
        return None
    if mask_now is not None and mask_now.dtype != torch.bool:
        raise ValueError("mask_now must be boolean")
    if mask_mem is not None and mask_mem.dtype != torch.bool:
        raise ValueError("mask_mem must be boolean")

    B_now = mask_now.shape[0] if mask_now is not None else None
    B_mem = mask_mem.shape[0] if mask_mem is not None else None
    if B_now is not None and B_mem is not None and B_now != B_mem:
        raise ValueError("mask_now and mask_mem must have same batch size when both provided")

    # If no current mask is provided, we cannot build [B,T_now,T_mem+T_now] reliably.
    # In that case, return None so callers can skip masking or build their own.
    if mask_now is None:
        return None

    B = mask_now.shape[0]
    T_now = mask_now.shape[1]

    T_mem = mask_mem.shape[1] if mask_mem is not None else 0

    # Key side mask: concat(mem_mask, now_mask)
    key_mask_mem = mask_mem if mask_mem is not None else torch.zeros(B, T_mem, dtype=torch.bool, device=mask_now.device)
    key_mask_now = mask_now
    key_mask = torch.cat([key_mask_mem, key_mask_now], dim=1)  # [B, T_mem+T_now]

    # Broadcast to [B, T_now, T_mem+T_now]
    attn_mask = key_mask.unsqueeze(1).expand(B, T_now, T_mem + T_now).clone()
    # If a query token itself is invalid, mask all its attends
    q_invalid = mask_now.unsqueeze(-1)  # [B, T_now, 1]
    attn_mask = attn_mask | q_invalid
    return attn_mask


def generate_rel_pos_bias(
    seq_len_q: int,
    seq_len_kv: int,
    d_model: int,
    max_rel: int = 4096,
) -> Tensor:
    """Generate a relative position "index" tensor for attention.

    Note: This helper returns relative offset indices of shape [seq_len_q, seq_len_kv]
    in the range [0, 2*max_rel], which can be embedded by an attention module that
    holds a learnable embedding table. We do not return learnable bias weights here
    to avoid global state.

    Offsets are defined with respect to the current frame tokens as queries (0..T_now-1)
    and the concatenated memory+current tokens as keys (0..T_mem+T_now-1). The relative
    offset for (i, j) is defined as (j - (seq_len_kv - seq_len_q) - i) and clamped.

    Args:
        seq_len_q: Current sequence length (T_now).
        seq_len_kv: Key/value sequence length (T_mem + T_now).
        d_model: Unused here, kept for API completeness.
        max_rel: Max absolute relative distance to clamp to.

    Returns:
        rel_indices: LongTensor of shape [seq_len_q, seq_len_kv] in [0, 2*max_rel].
    """
    # j index includes memory then current; shift by (seq_len_kv - seq_len_q) to align current segment.
    device = torch.device("cpu")
    i = torch.arange(seq_len_q, dtype=torch.long, device=device).unsqueeze(1)  # [T_now,1]
    j = torch.arange(seq_len_kv, dtype=torch.long, device=device).unsqueeze(0)  # [1,T_mem+T_now]
    rel = j - (seq_len_kv - seq_len_q) - i  # [T_now, T_mem+T_now]
    rel = torch.clamp(rel, min=-max_rel, max=max_rel)
    rel = rel + max_rel  # shift to [0, 2*max_rel]
    return rel


# ------------------------------
# Relative-position Multi-Head Attention with learnable bias
# ------------------------------

class RelPosMultiHeadAttn(nn.Module):
    """Relative positional multi-head attention with optional dropout and masking.

    This implementation uses a learnable per-head bias table over relative offsets
    (clamped range [-max_rel, max_rel]) instead of the full TXL relative shift trick.
    It adds the bias directly to attention logits for numerical stability and simplicity.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, max_rel: int = 4096) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.max_rel = max_rel

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable per-head bias embedding over relative distances
        self.rel_bias = nn.Embedding(2 * max_rel + 1, nhead)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x_q: Tensor,           # [B, T_now, C]
        k_cat: Tensor,         # [B, T_mem+T_now, C]
        v_cat: Tensor,         # [B, T_mem+T_now, C]
        r_rel: Tensor,         # [T_now, T_mem+T_now] relative offset indices
        attn_mask: Optional[Tensor] = None,  # [B, T_now, T_mem+T_now] (True => mask)
    ) -> Tensor:
        """Compute attention output for the current sequence.

        Returns:
            y: [B, T_now, C]
        """
        B, T_now, C = x_q.shape
        if k_cat.shape[0] != B or v_cat.shape[0] != B:
            raise ValueError("Batch size mismatch among x_q, k_cat, v_cat")
        if k_cat.shape[2] != C or v_cat.shape[2] != C:
            raise ValueError("Channel dim C mismatch among x_q, k_cat, v_cat")
        T_kv = k_cat.shape[1]
        if r_rel.shape != (T_now, T_kv):
            raise ValueError(f"r_rel must be [T_now, T_mem+T_now], got {tuple(r_rel.shape)}")
        if attn_mask is not None and attn_mask.shape != (B, T_now, T_kv):
            raise ValueError(f"attn_mask must be [B, T_now, T_mem+T_now], got {tuple(attn_mask.shape)}")

        # Project to Q, K, V and split heads
        q = self.q_proj(x_q)  # [B, T_now, C]
        k = self.k_proj(k_cat)  # [B, T_kv, C]
        v = self.v_proj(v_cat)  # [B, T_kv, C]

        q = q.view(B, T_now, self.nhead, self.d_head).transpose(1, 2)  # [B, H, Tq, Dh]
        k = k.view(B, T_kv, self.nhead, self.d_head).transpose(1, 2)   # [B, H, Tk, Dh]
        v = v.view(B, T_kv, self.nhead, self.d_head).transpose(1, 2)   # [B, H, Tk, Dh]

        # Content-based attention logits
        # logits = (Q * K^T) / sqrt(d_head)
        logits = torch.matmul(q.to(dtype=torch.float32), k.transpose(-1, -2).to(dtype=torch.float32))  # [B,H,Tq,Tk]
        logits = logits * self.scale

        # Add relative positional bias per head
        # r_rel: [Tq, Tk] long indices -> [Tq,Tk,H] via embedding, then permute to [H,Tq,Tk]
        if r_rel.dtype != torch.long:
            raise ValueError("r_rel must be LongTensor of relative offset indices")
        rel_bias = self.rel_bias(r_rel)  # [Tq, Tk, H]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, Tq, Tk]
        logits = logits + rel_bias

        # Apply attention mask
        if attn_mask is not None:
            # True => mask; set to large negative
            logits = logits.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        y = torch.matmul(attn.to(dtype=v.dtype), v)  # [B,H,Tq,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T_now, C)  # [B,Tq,C]
        y = self.o_proj(y)
        return y


# ------------------------------
# TXL Block with memory
# ------------------------------

class TXLBlock(nn.Module):
    """One TXL block: Pre-LN -> RelPosAttn (with mem) -> residual; Pre-LN -> FFN -> residual.

    Memory policy: Concatenate provided memory with current tokens to build K/V. New memory is
    formed by concatenating old memory with current output, then truncating to the most recent
    mem_len_tokens. Memory is detached from the graph before being returned.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        ffn_hidden: int = 512,
        dropout: float = 0.0,
        mem_len_tokens: int = 2048,
        max_rel: int = 4096,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mem_len_tokens = mem_len_tokens

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RelPosMultiHeadAttn(d_model=d_model, nhead=nhead, dropout=dropout, max_rel=max_rel)
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.max_rel = max_rel

    def _build_kv(self, x_preln: Tensor, mem: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if mem is None:
            return x_preln, x_preln
        if mem.shape[0] != x_preln.shape[0] or mem.shape[2] != x_preln.shape[2]:
            raise ValueError("mem must be [B,T_mem,C] matching batch and channel")
        mem_preln = self.ln1(mem)
        k_cat = torch.cat([mem_preln, x_preln], dim=1)
        v_cat = torch.cat([mem_preln, x_preln], dim=1)
        return k_cat, v_cat

    def forward(
        self,
        x_now: Tensor,                  # [B, T_now, C]
        mem: Optional[Tensor] = None,   # [B, T_mem, C]
        attn_mask: Optional[Tensor] = None,  # [B, T_now, T_mem+T_now]
    ) -> Tuple[Tensor, Tensor]:
        if x_now.ndim != 3:
            raise ValueError(f"x_now must be [B,T_now,C], got {tuple(x_now.shape)}")
        B, T_now, C = x_now.shape
        if C != self.d_model:
            raise ValueError(f"x_now C={C} != d_model={self.d_model}")
        if mem is not None and mem.ndim != 3:
            raise ValueError("mem must be [B,T_mem,C] or None")
        if attn_mask is not None and attn_mask.ndim != 3:
            raise ValueError("attn_mask must be [B,T_now,T_mem+T_now] or None")

        # Pre-LN and Attention
        q_in = self.ln1(x_now)
        k_cat, v_cat = self._build_kv(q_in, mem)
        T_kv = k_cat.shape[1]

        # Relative indices for this block
        r_rel = generate_rel_pos_bias(seq_len_q=T_now, seq_len_kv=T_kv, d_model=self.d_model, max_rel=self.max_rel)
        r_rel = r_rel.to(q_in.device)

        y = self.attn(q_in, k_cat, v_cat, r_rel=r_rel, attn_mask=attn_mask)
        x = x_now + self.drop1(y)

        # Pre-LN and FFN
        z_in = self.ln2(x)
        z = self.ffn(z_in)
        y_now = x + self.drop2(z)

        # Update memory: concat old mem with current output, then keep most recent mem_len_tokens
        if mem is None:
            new_mem = y_now.detach()
        else:
            new_mem = torch.cat([mem, y_now], dim=1).detach()
        if self.mem_len_tokens > 0 and new_mem.shape[1] > self.mem_len_tokens:
            new_mem = new_mem[:, -self.mem_len_tokens :, :]

        return y_now, new_mem


# ------------------------------
# TXL Backbone (stacked blocks)
# ------------------------------

class TXLBackbone(nn.Module):
    """Stacked TXL blocks over fused per-frame tokens, with per-layer memory.

    The module processes a single frame's fused tokens [B, 1+N, C] at a time, with each layer
    receiving and returning its own memory. Memory length is configured in frames but converted
    to tokens by multiplying with tokens_per_frame.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_hidden: int = 512,
        dropout: float = 0.0,
        mem_len_frames: int = 128,
        tokens_per_frame: int = 17,
        max_rel: int = 4096,
    ) -> None:
        super().__init__()
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        mem_len_tokens = mem_len_frames * tokens_per_frame
        self.tokens_per_frame = tokens_per_frame

        self.layers = nn.ModuleList(
            [
                TXLBlock(
                    d_model=d_model,
                    nhead=nhead,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    mem_len_tokens=mem_len_tokens,
                    max_rel=max_rel,
                )
                for _ in range(num_layers)
            ]
        )
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(
        self,
        t_prop: Tensor,                 # [B, C]
        t_vis: Tensor,                  # [B, N, C]
        mems: Optional[List[Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,  # [B, T_now, T_mem+T_now]
    ) -> Tuple[Tensor, List[Tensor]]:
        if t_prop.ndim != 2:
            raise ValueError(f"t_prop must be [B,C], got {tuple(t_prop.shape)}")
        if t_vis.ndim != 3:
            raise ValueError(f"t_vis must be [B,N,C], got {tuple(t_vis.shape)}")
        if t_prop.shape[0] != t_vis.shape[0] or t_prop.shape[1] != t_vis.shape[2]:
            raise ValueError("Batch or channel mismatch between t_prop and t_vis")

        x_now = fuse_tokens(t_prop, t_vis, prop_first=True)  # [B, 1+N, C]
        B, T_now, C = x_now.shape

        if mems is None:
            mems = [None for _ in range(self.num_layers)]
        if len(mems) != self.num_layers:
            raise ValueError(f"mems must have length {self.num_layers}")

        new_mems: List[Tensor] = []
        h = x_now
        for i, (layer, mem) in enumerate(zip(self.layers, mems)):
            y, new_mem = layer(h, mem=mem, attn_mask=attn_mask)
            new_mems.append(new_mem)
            h = y
        # h is [B, 1+N, C]
        return h, new_mems


# ------------------------------
# Pooling and Heads
# ------------------------------

def modality_pool(x_seq: Tensor) -> Tensor:
    """Pool fused tokens by taking proprio token (idx 0) and mean over visual tokens.

    Args:
        x_seq: [B, 1+N, C]

    Returns:
        z: [B, 2C] = concat([x[:,0,:], mean(x[:,1:,:])])
    """
    if x_seq.ndim != 3:
        raise ValueError(f"x_seq must be [B,1+N,C], got {tuple(x_seq.shape)}")
    B, T, C = x_seq.shape
    if T < 2:
        raise ValueError("x_seq must contain at least one proprio and one visual token (T>=2)")
    prop = x_seq[:, 0, :]                   # [B, C]
    vis_mean = x_seq[:, 1:, :].mean(dim=1)  # [B, C]
    return torch.cat([prop, vis_mean], dim=-1)  # [B, 2C]


class PolicyHeads(nn.Module):
    """Simple policy/value heads operating on modality-wise pooled features."""

    def __init__(self, c: int = 128, action_dim: int = 12) -> None:
        super().__init__()
        self.actor = nn.Sequential(nn.LayerNorm(2 * c), nn.Linear(2 * c, action_dim))
        self.critic = nn.Sequential(nn.LayerNorm(2 * c), nn.Linear(2 * c, 1))

    def forward(self, x_seq: Tensor) -> Tuple[Tensor, Tensor]:
        z = modality_pool(x_seq)
        return self.actor(z), self.critic(z)


# ------------------------------
# Demo
# ------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, N = 2, 128, 16
    TPF = 1 + N

    # Random inputs simulating encoders' outputs
    t_prop = torch.randn(B, C)
    t_vis = torch.randn(B, N, C)

    txl = TXLBackbone(d_model=C, nhead=4, num_layers=3, ffn_hidden=512, dropout=0.0, mem_len_frames=8, tokens_per_frame=TPF)
    heads = PolicyHeads(c=C, action_dim=12)

    mems: List[Optional[Tensor]] = [None] * 3

    steps = 5
    for step in range(steps):
        x_seq, mems = txl(t_prop, t_vis, mems=mems, attn_mask=None)
        logits, value = heads(x_seq)
        print(f"x_seq: {tuple(x_seq.shape)}, logits: {tuple(logits.shape)}, value: {tuple(value.shape)}")
        for i, m in enumerate(mems):
            assert m is not None
            print(f"mem[{i}] shape: {tuple(m.shape)} (<= {8*TPF} tokens)")
