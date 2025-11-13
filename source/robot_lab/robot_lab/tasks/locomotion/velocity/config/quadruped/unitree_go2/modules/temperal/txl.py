"""Lightweight Transformer-XL style temporal module optimized for continuous embeddings."""

from typing import List, Optional, Tuple

import torch


_KAIMING_A = float(torch.sqrt(torch.tensor(5.0)))


class TransformerXLTemporal(torch.nn.Module):
    """Temporal encoder stack with Transformer-XL style memory."""

    def __init__(
        self,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        d_inner: int = 256,
        mem_len: int = 64,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        norm_first: bool = True,
        clamp_len: Optional[int] = None,
        use_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if mem_len < 0:
            raise ValueError("mem_len must be non-negative")

        self.d_model = d_model
        self.n_layer = n_layer
        self.mem_len = mem_len
        self.use_rel_pos = use_rel_pos

        self.layers = torch.nn.ModuleList(
            [
                _TransformerXLLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_inner=d_inner,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_first=norm_first,
                    mem_len=mem_len,
                    clamp_len=clamp_len,
                    use_rel_pos=use_rel_pos,
                )
                for _ in range(n_layer)
            ]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=_KAIMING_A)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mems: Optional[List[Optional[torch.Tensor]]] = None,
        causal_mask: bool = True,
        return_mems: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run the temporal encoder with optional recurrent memory."""

        if x.dim() != 3:
            raise ValueError("Input x must have shape [B, S, C]")
        batch, seq_len, channels = x.shape
        if channels != self.d_model:
            raise ValueError(f"Expected input embedding dim {self.d_model}, got {channels}")

        if mems is None:
            mems = [None] * self.n_layer
        if len(mems) != self.n_layer:
            raise ValueError("mems length must match number of layers")

        new_mems: List[torch.Tensor] = []
        h = x
        for layer, mem in zip(self.layers, mems):
            if mem is not None and (mem.dim() != 3 or mem.size(0) != batch or mem.size(2) != self.d_model):
                raise ValueError("Each memory must be [B, M, d_model]")
            h, layer_mem = layer(h, mem, causal_mask)
            if return_mems:
                new_mems.append(layer_mem)

        return h, (new_mems if return_mems else None)

    def reset_mems(self, batch_size: int) -> List[torch.Tensor]:
        """Return zero-length memories with the correct dtype/device."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        empty = torch.empty(batch_size, 0, self.d_model, device=device, dtype=dtype)
        return [empty.clone() for _ in range(self.n_layer)]

    @staticmethod
    def detach_mems(mems: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        """Detach memories from the computation graph."""

        return [m.detach() if m is not None else None for m in mems]

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters."""

        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(int(p.numel()) for p in params)


class _TransformerXLLayer(torch.nn.Module):
    """Single Transformer-XL style layer."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_inner: int,
        dropout: float,
        attn_dropout: float,
        norm_first: bool,
        mem_len: int,
        clamp_len: Optional[int],
        use_rel_pos: bool,
    ) -> None:
        super().__init__()
        self.mem_len = mem_len
        self.norm_first = norm_first
        self.dropout = torch.nn.Dropout(dropout)
        self.attn = _TemporalMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            attn_dropout=attn_dropout,
            use_rel_pos=use_rel_pos,
            clamp_len=clamp_len,
        )
        self.ff = _PositionwiseFFN(d_model=d_model, d_inner=d_inner, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor],
        causal_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        mem = self._ensure_mem(x, mem)
        mem_len = mem.size(1)

        if self.norm_first:
            cat = torch.cat([mem, x], dim=1) if mem_len > 0 else x
            cat_norm = self.norm1(cat)
            kv_input = cat_norm
            q_input = cat_norm[:, -seq_len:, :]
        else:
            kv_input = torch.cat([mem, x], dim=1) if mem_len > 0 else x
            q_input = x

        attn_out = self.attn(
            q_input=q_input,
            kv_input=kv_input,
            mem_len=mem_len,
            causal_mask=causal_mask,
        )

        if self.norm_first:
            h = x + self.dropout(attn_out)
            ff_input = self.norm2(h)
            h = h + self.ff(ff_input)
        else:
            h = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ff(h)
            h = self.norm2(h + ff_out)

        new_mem = self._update_mem(mem, h)
        return h, new_mem

    def _ensure_mem(self, x: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        if mem is None:
            return x.new_empty(x.size(0), 0, x.size(2))
        return mem

    def _update_mem(self, mem: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.mem_len == 0:
            return h.new_empty(h.size(0), 0, h.size(2))
        cat = torch.cat([mem, h], dim=1) if mem.size(1) > 0 else h
        if cat.size(1) > self.mem_len:
            cat = cat[:, -self.mem_len :, :]
        return cat.detach()


class _TemporalMultiHeadAttention(torch.nn.Module):
    """Multi-head attention with optional Transformer-XL relative positions."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_dropout: float,
        use_rel_pos: bool,
        clamp_len: Optional[int],
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim**-0.5
        self.use_rel_pos = use_rel_pos
        self.clamp_len = clamp_len

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = torch.nn.Dropout(attn_dropout)
        self.register_buffer("_pos_cache", torch.zeros(0, self.head_dim), persistent=False)
        self._cache_device: Optional[torch.device] = None
        self._cache_dtype: Optional[torch.dtype] = None

        if use_rel_pos:
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.u_bias = torch.nn.Parameter(torch.zeros(n_head, self.head_dim))
        else:
            self.register_buffer("inv_freq", torch.empty(0), persistent=False)
            self.u_bias = torch.nn.Parameter(torch.zeros(n_head, self.head_dim))

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        mem_len: int,
        causal_mask: bool,
    ) -> torch.Tensor:
        batch, q_len, _ = q_input.shape
        k_len = kv_input.size(1)

        q = self._shape(self.q_proj(q_input))
        k = self._shape(self.k_proj(kv_input))
        v = self._shape(self.v_proj(kv_input))

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self.use_rel_pos:
            pos_emb = self._get_positional_embedding(k_len, q_input.device, q_input.dtype)
            rel_term = self._relative_attention(q, pos_emb, q_len, k_len)
            attn_scores = attn_scores + rel_term

        attn_scores = attn_scores * self.scale
        mask = self._build_causal_mask(q_len, k_len, mem_len, q_input.device, causal_mask)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_prob = torch.softmax(attn_scores, dim=-1)
        attn_prob = self.attn_dropout(attn_prob)

        context = torch.matmul(attn_prob, v)
        context = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.out_proj(context)

    def _shape(self, proj: torch.Tensor) -> torch.Tensor:
        batch, length, _ = proj.shape
        proj = proj.view(batch, length, self.n_head, self.head_dim)
        return proj.transpose(1, 2).contiguous()

    def _get_positional_embedding(
        self, k_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if k_len == 0:
            return self._pos_cache.new_empty(0, self.head_dim)
        if (
            self._pos_cache.size(0) >= k_len
            and self._cache_device == device
            and self._cache_dtype == dtype
        ):
            return self._pos_cache[-k_len:]

        pos_seq = torch.arange(k_len - 1, -1, -1, device=device, dtype=dtype)
        if self.clamp_len is not None:
            pos_seq = pos_seq.clamp(max=self.clamp_len - 1)
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if pos_emb.size(1) < self.head_dim:
            pad = self.head_dim - pos_emb.size(1)
            pos_emb = torch.nn.functional.pad(pos_emb, (0, pad))
        elif pos_emb.size(1) > self.head_dim:
            pos_emb = pos_emb[:, : self.head_dim]

        self._pos_cache = pos_emb.detach()
        self._cache_device = device
        self._cache_dtype = dtype
        return pos_emb

    def _relative_attention(
        self, q: torch.Tensor, pos_emb: torch.Tensor, q_len: int, k_len: int
    ) -> torch.Tensor:
        if k_len == 0 or pos_emb.numel() == 0:
            return q.new_zeros(q.size(0), self.n_head, q_len, k_len)
        r = pos_emb.unsqueeze(0).unsqueeze(0).expand(q.size(0), self.n_head, k_len, self.head_dim)
        q_bias = q + self.u_bias.view(1, self.n_head, 1, self.head_dim)
        rel_scores = torch.einsum("bhqd,bhld->bhql", q_bias, r)
        rel_scores = self._rel_shift(rel_scores, q_len, k_len)
        return rel_scores

    @staticmethod
    def _rel_shift(x: torch.Tensor, q_len: int, k_len: int) -> torch.Tensor:
        zero_pad = torch.zeros((*x.shape[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), k_len + 1, q_len)
        x = x_padded[:, :, 1:].view(x.size(0), x.size(1), q_len, k_len)
        return x

    def _build_causal_mask(
        self,
        q_len: int,
        k_len: int,
        mem_len: int,
        device: torch.device,
        causal_mask: bool,
    ) -> Optional[torch.Tensor]:
        if not causal_mask:
            return None
        if k_len == 0:
            return None
        query_positions = torch.arange(q_len, device=device).unsqueeze(1) + mem_len
        key_positions = torch.arange(k_len, device=device).unsqueeze(0)
        mask = key_positions > query_positions
        return mask.unsqueeze(0).unsqueeze(0)


class _PositionwiseFFN(torch.nn.Module):
    """Feed-forward block used in each encoder layer."""

    def __init__(self, d_model: int, d_inner: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_inner)
        self.linear2 = torch.nn.Linear(d_inner, d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TransformerXLTemporal(
        d_model=128,
        n_layer=2,
        n_head=4,
        d_inner=256,
        mem_len=32,
    )
    x = torch.randn(2, 5, 128)
    y, mems1 = model(x)
    assert y.shape == (2, 5, 128), "Unexpected output shape"
    assert mems1 is not None and len(mems1) == 2
    for mem in mems1:
        assert mem.shape == (2, min(5, 32), 128)

    x2 = torch.randn(2, 3, 128)
    y2, mems2 = model(x2, mems=mems1)
    assert y2.shape == (2, 3, 128)
    assert mems2 is not None
    for mem in mems2:
        assert mem.shape == (2, min(8, 32), 128)
        assert torch.isfinite(mem).all()

    assert torch.isfinite(y).all() and torch.isfinite(y2).all()
    print("TransformerXLTemporal smoke tests passed.")
