import os
import sys
from typing import List, Tuple

import torch


def _ensure_modules_on_path() -> None:
    """Ensure the project root containing 'modules' is available on sys.path."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    curr = file_dir
    while True:
        modules_dir = os.path.join(curr, "modules")
        if os.path.isdir(modules_dir):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            return
        parent = os.path.dirname(curr)
        if parent == curr:
            raise RuntimeError("Unable to locate a project root containing 'modules'.")
        curr = parent


_ensure_modules_on_path()

from modules.encoders.fusion_transformer import MultiModalFusionTransformer
from modules.temperal.txl import TransformerXLTemporal
from modules.tokenizers.depth_encoder import DepthEncoder
from modules.tokenizers.proprio_encoder import ProprioEncoder

import time


def _count_params(module, trainable_only: bool = True) -> int:
    params = module.parameters() if not trainable_only else (p for p in module.parameters() if p.requires_grad)
    return sum(int(p.numel()) for p in params)


def _fmt_int(n: int) -> str:
    # human-friendly, with underscores and M suffix when large
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M ({n:_d})"
    if n >= 1_000:
        return f"{n/1_000:.3f}K ({n:_d})"
    return f"{n:_d}"


def run_full_pipeline_smoketest() -> None:
    """Run a perception-to-temporal forward smoketest for the locomotion stack."""
    torch.manual_seed(0)

    B = 2
    S = 5
    state_dim = 31
    hist_len = 3
    in_frames = 4
    in_size = 64
    token_dim = 128
    grid_size = 4
    mem_len = 16

    prop_enc = ProprioEncoder(state_dim=state_dim, hist_len=hist_len, token_dim=token_dim)
    depth_enc = DepthEncoder(
        in_frames=in_frames,
        in_size=in_size,
        token_dim=token_dim,
        grid_size=grid_size,
    )
    fusion = MultiModalFusionTransformer(token_dim=token_dim, num_layers=2, num_heads=4)
    txl = TransformerXLTemporal(
        d_model=token_dim,
        n_layer=2,
        n_head=4,
        d_inner=256,
        mem_len=mem_len,
    )

    proprios = torch.randn(B, S, hist_len * state_dim)
    depths = torch.randn(B, S, in_frames, in_size, in_size)

    hidden_chunks: List[torch.Tensor] = []
    for t in range(S):
        prop_t = proprios[:, t, :]
        depth_t = depths[:, t, :, :, :]

        t_prop = prop_enc(prop_t)
        t_vis = depth_enc(depth_t)
        fused = fusion(t_prop, t_vis)

        assert "all_pooled" in fused, "Fusion output missing 'all_pooled'."
        hidden_t = fused["all_pooled"]
        assert hidden_t.shape == (B, token_dim), (
            f"Unexpected fused token shape {tuple(hidden_t.shape)}; expected {(B, token_dim)}."
        )
        hidden_chunks.append(hidden_t.unsqueeze(1))

    hidden_seq = torch.cat(hidden_chunks, dim=1)
    assert hidden_seq.shape == (B, S, token_dim), (
        f"Hidden sequence shape {tuple(hidden_seq.shape)} != {(B, S, token_dim)}."
    )
    assert torch.isfinite(hidden_seq).all(), "NaN or Inf detected in hidden sequence."

    y, mems = txl(hidden_seq, mems=None, causal_mask=True, return_mems=True)
    assert y.shape == (B, S, token_dim), f"TXL output shape {tuple(y.shape)} invalid."
    assert isinstance(mems, list) and len(mems) == txl.n_layer, (
        "TransformerXLTemporal must return mems per layer."
    )

    for idx, mem in enumerate(mems):
        assert mem.ndim == 3, f"Memory tensor at layer {idx} must be 3D."
        bsz, mem_steps, dim = mem.shape
        assert bsz == B and dim == token_dim, (
            f"Memory tensor {idx} shape {tuple(mem.shape)} inconsistent with batch/token dims."
        )
        assert 0 < mem_steps <= mem_len, (
            f"Memory tensor {idx} length {mem_steps} exceeds configured mem_len={mem_len}."
        )
        assert torch.isfinite(mem).all(), f"NaN/Inf detected in memory tensor {idx}."

    assert torch.isfinite(y).all(), "NaN or Inf detected in TXL output."

    x1 = hidden_seq[:, :3, :]
    x2 = hidden_seq[:, 3:, :]
    y1, mems1 = txl(x1, mems=None, causal_mask=True, return_mems=True)
    y2, mems2 = txl(x2, mems=mems1, causal_mask=True, return_mems=True)

    assert y1.shape == (B, 3, token_dim) and y2.shape == (B, 2, token_dim), (
        "Chunked TXL outputs have incorrect shapes."
    )
    for idx, mem in enumerate(mems1 + mems2):
        assert mem.shape[0] == B and mem.shape[2] == token_dim, (
            f"Chunked memory tensor {idx} has invalid shape {tuple(mem.shape)}."
        )
        assert mem.shape[1] <= mem_len, (
            f"Chunked memory tensor {idx} length {mem.shape[1]} exceeds mem_len={mem_len}."
        )

    print(
        "Full pipeline smoketest passed:",
        f"hidden_seq={tuple(hidden_seq.shape)}",
        f"y={tuple(y.shape)}",
    )

    # Parameter counting summary
    p_prop = _count_params(prop_enc)
    p_depth = _count_params(depth_enc)
    p_fuse = _count_params(fusion)
    p_txl = _count_params(txl)
    p_total = p_prop + p_depth + p_fuse + p_txl
    print(
        "Params:",
        f"ProprioEncoder={_fmt_int(p_prop)}",
        f"DepthEncoder={_fmt_int(p_depth)}",
        f"FusionTransformer={_fmt_int(p_fuse)}",
        f"TXL={_fmt_int(p_txl)}",
        f"Total={_fmt_int(p_total)}",
    )


def run_gradient_and_perf_smoketest() -> None:
    """Run tiny gradient/backprop sanity and quick perf timings."""
    torch.manual_seed(0)

    # Small config to keep test quick
    B = 2
    S = 4
    state_dim = 31
    hist_len = 3
    in_frames = 4
    in_size = 64
    token_dim = 128
    grid_size = 4
    mem_len = 8

    # Build modules
    prop_enc = ProprioEncoder(state_dim=state_dim, hist_len=hist_len, token_dim=token_dim)
    depth_enc = DepthEncoder(
        in_frames=in_frames,
        in_size=in_size,
        token_dim=token_dim,
        grid_size=grid_size,
    )
    fusion = MultiModalFusionTransformer(token_dim=token_dim, num_layers=2, num_heads=4)
    txl = TransformerXLTemporal(
        d_model=token_dim,
        n_layer=2,
        n_head=4,
        d_inner=256,
        mem_len=mem_len,
    )

    modules = [prop_enc, depth_enc, fusion, txl]
    for m in modules:
        m.train()

    # Dummy data
    proprios = torch.randn(B, S, hist_len * state_dim)
    depths = torch.randn(B, S, in_frames, in_size, in_size)

    # Build a very small optimizer over all params
    all_params = []
    for m in modules:
        all_params += list(m.parameters())
    opt = torch.optim.SGD(all_params, lr=1e-3, momentum=0.0)

    def fwd_once() -> torch.Tensor:
        hidden_chunks = []
        for t in range(S):
            prop_t = proprios[:, t, :]
            depth_t = depths[:, t, :, :, :]
            t_prop = prop_enc(prop_t)
            t_vis = depth_enc(depth_t)
            fused = fusion(t_prop, t_vis)
            hidden_chunks.append(fused["all_pooled"].unsqueeze(1))
        hidden_seq = torch.cat(hidden_chunks, dim=1)
        y, _ = txl(hidden_seq, mems=None, causal_mask=True, return_mems=True)
        return y

    # Backprop/stability: minimize ||y||^2, check grad finiteness and some decrease
    with torch.enable_grad():
        y0 = fwd_once()
        loss0 = (y0.pow(2).mean())
        opt.zero_grad(set_to_none=True)
        loss0.backward()

        # gradient checks
        total_grad_norm = 0.0
        nonzero_grads = 0
        for p in all_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            if torch.isfinite(g).all():
                gn = g.norm().item()
                total_grad_norm += gn
                if gn > 0:
                    nonzero_grads += 1
            else:
                raise AssertionError("Non-finite gradient detected")
        assert nonzero_grads > 0, "All gradients are zero."
        assert total_grad_norm < 1e6, f"Gradient norm too large: {total_grad_norm}"

        # one optimizer step to see loss change
        opt.step()
        y1 = fwd_once().detach()
        loss1 = (y1.pow(2).mean()).item()
        print(
            "Grad OK:",
            f"loss0={loss0.item():.6f}",
            f"loss1={loss1:.6f}",
            f"total_grad_norm={total_grad_norm:.3f}",
            f"nonzero_grads={nonzero_grads}",
        )

    # Quick perf timing (CPU). Keep iterations small to stay fast.
    iters = 5
    # warmup
    for _ in range(2):
        _ = fwd_once()
    # time tokenization+fusion per-step
    start = time.perf_counter()
    for _ in range(iters):
        for t in range(S):
            _ = fusion(prop_enc(proprios[:, t, :]), depth_enc(depths[:, t]))
    tok_fuse_ms = (time.perf_counter() - start) * 1000.0 / (iters * S)

    # time full pipeline per-sequence (S steps + TXL)
    start = time.perf_counter()
    for _ in range(iters):
        _ = fwd_once()
    full_seq_ms = (time.perf_counter() - start) * 1000.0 / iters

    print(
        "Perf:",
        f"tokenize+fuse per-step ~ {tok_fuse_ms:.2f} ms",
        f"full pipeline per-seq (S={S}) ~ {full_seq_ms:.2f} ms",
    )


if __name__ == "__main__":
    run_full_pipeline_smoketest()
    run_gradient_and_perf_smoketest()
