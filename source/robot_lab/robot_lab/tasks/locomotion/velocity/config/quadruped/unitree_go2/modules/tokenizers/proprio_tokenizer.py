from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class ProprioTokenizerCfg:
    """Configuration for the proprioception tokenizer."""

    proprio_keys: Tuple[str, ...] = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    use_trig_for_joint_pos: bool = False
    zscore_mean: Optional[np.ndarray] = None
    zscore_std: Optional[np.ndarray] = None
    clip_val: float = 10.0
    hidden_dims: Tuple[int, ...] = (256, 256)
    d_model: int = 512
    num_tokens: int = 4
    attn_heads: int = 4
    freeze_frontend: bool = True
    require_strict_pretrained: bool = True
    verbose: bool = True


class PretrainedFrontEnd(nn.Module):
    """MLP front-end that can load linear layers from a teacher checkpoint."""

    _KEY_TOKENS = ("actor", "policy", "pi", "mlp")

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        self.linears = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
            prev_dim = dim
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        for linear in self.linears:
            x = self.activation(linear(x))
        return x

    def _load_state_dict_obj(self, ckpt: Union[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if isinstance(ckpt, str):
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")
            state = torch.load(ckpt, map_location="cpu")
        else:
            state = ckpt

        if not isinstance(state, dict):
            raise TypeError("Checkpoint must be a mapping containing tensors.")

        # Dig into common wrappers.
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        return cast(Dict[str, Tensor], state)

    def load_actor_front_from_teacher(
        self,
        ckpt: Union[str, Dict[str, Tensor]],
        verbose: bool = True,
    ) -> Tuple[bool, int]:
        """Attempt to copy linear weights from a teacher policy checkpoint."""

        state = self._load_state_dict_obj(ckpt)
        candidates: List[Tuple[str, Optional[str]]] = []

        for key in state.keys():
            lower = key.lower()
            if "weight" not in lower and "bias" not in lower:
                continue
            if any(token in lower for token in ("lstm", "gru")):
                continue
            if not any(token in lower for token in self._KEY_TOKENS):
                continue
            if lower.endswith("weight"):
                base = key[: -len("weight")].rstrip(".")
                bias_key = f"{base}.bias" if f"{base}.bias" in state else None
                candidates.append((key, bias_key))

        copied = 0
        candidate_idx = 0
        total_candidates = len(candidates)
        for layer_idx, linear in enumerate(self.linears):
            matched = False
            while candidate_idx < total_candidates:
                weight_key, bias_key = candidates[candidate_idx]
                candidate_idx += 1
                weight_tensor = state[weight_key]
                if weight_tensor.ndim != 2 or weight_tensor.shape != linear.weight.data.shape:
                    continue
                bias_tensor = state.get(bias_key) if bias_key is not None else None
                if bias_tensor is not None and bias_tensor.shape != linear.bias.data.shape:
                    continue
                linear.weight.data.copy_(weight_tensor)
                if bias_tensor is not None:
                    linear.bias.data.copy_(bias_tensor)
                matched = True
                copied += 1
                if verbose:
                    print(
                        f"[ProprioTokenizer] Copied {weight_key} -> frontend.linear[{layer_idx}]"
                        f" (bias: {bias_key if bias_tensor is not None else 'none'})"
                    )
                break
            if not matched and verbose:
                print(
                    f"[ProprioTokenizer] No matching pretrained layer for frontend.linear[{layer_idx}] "
                    f"(in_features={linear.in_features}, out_features={linear.out_features})"
                )

        success = copied > 0
        if verbose:
            print(f"[ProprioTokenizer] Pretrained front-end copy result: success={success}, layers={copied}")
        return success, copied


class TokenHead(nn.Module):
    """Cross-attention token head producing tokens and pooled embedding."""

    def __init__(self, d_model: int, num_tokens: int, attn_heads: int) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if d_model % attn_heads != 0:
            raise ValueError("d_model must be divisible by attn_heads.")

        self.d_model = d_model
        self.num_tokens = num_tokens

        self.slot_proj = nn.Linear(d_model, num_tokens * d_model)
        nn.init.xavier_uniform_(self.slot_proj.weight)
        nn.init.zeros_(self.slot_proj.bias)

        self.token_queries = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, attn_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        ff_hidden = max(d_model * 4, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, d_model] input, got shape {tuple(x.shape)}")
        batch = x.shape[0]
        slots = self.slot_proj(x).view(batch, self.num_tokens, self.d_model)

        queries = self.token_queries.unsqueeze(0).expand(batch, -1, -1)
        attn_out, _ = self.attn(
            self.norm_q(queries),
            self.norm_slots(slots),
            self.norm_slots(slots),
            need_weights=False,
        )
        tokens = queries + attn_out
        tokens = tokens + self.ff(self.norm_ff(tokens))
        tokens = self.output_norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled


class ProprioTokenizer(nn.Module):
    """Tokenizer turning proprioceptive observations into transformer tokens."""

    def __init__(self, cfg: Optional[ProprioTokenizerCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or ProprioTokenizerCfg()
        self.frontend: Optional[PretrainedFrontEnd] = None
        self.projector: Optional[nn.Linear] = None
        self.token_head: Optional[TokenHead] = None
        self.register_buffer("_zscore_mean", torch.tensor([]), persistent=False)
        self.register_buffer("_zscore_std", torch.tensor([]), persistent=False)
        self._built = False

    def _concatenate_obs(self, obs: Dict[str, Tensor]) -> Tensor:
        parts: List[Tensor] = []
        trig_applied = False
        for key in self.cfg.proprio_keys:
            if key not in obs:
                raise KeyError(f"Missing proprio key '{key}' in observation dict.")
            value = obs[key]
            if value.ndim != 2:
                raise ValueError(f"Observation '{key}' must have shape [B, dim], got {tuple(value.shape)}.")
            if key == "joint_pos" and self.cfg.use_trig_for_joint_pos:
                parts.append(torch.sin(value))
                parts.append(torch.cos(value))
                trig_applied = True
            parts.append(value)
        if not parts:
            raise ValueError("No observation tensors collected.")
        concatenated = torch.cat(parts, dim=-1)
        if self.cfg.use_trig_for_joint_pos and not trig_applied:
            raise KeyError("use_trig_for_joint_pos=True but 'joint_pos' key not provided.")
        return concatenated

    def _normalize(self, x: Tensor) -> Tensor:
        if self._zscore_mean.numel() == 0 or self._zscore_std.numel() == 0:
            return torch.clamp(x, min=-self.cfg.clip_val, max=self.cfg.clip_val)
        mean = self._zscore_mean.to(device=x.device, dtype=x.dtype)
        std = self._zscore_std.to(device=x.device, dtype=x.dtype)
        if (std == 0).any():
            raise ValueError("zscore_std contains zeros, cannot normalize.")
        x = (x - mean) / std
        return torch.clamp(x, min=-self.cfg.clip_val, max=self.cfg.clip_val)

    def _maybe_build(self, input_dim: int) -> None:
        if self._built:
            return

        self.frontend = PretrainedFrontEnd(input_dim, self.cfg.hidden_dims)
        if self.cfg.freeze_frontend:
            for param in self.frontend.parameters():
                param.requires_grad = False

        last_hidden = self.cfg.hidden_dims[-1]
        if last_hidden != self.cfg.d_model:
            self.projector = nn.Linear(last_hidden, self.cfg.d_model)
        else:
            self.projector = None

        self.token_head = TokenHead(self.cfg.d_model, self.cfg.num_tokens, self.cfg.attn_heads)

        if self.cfg.zscore_mean is not None and self.cfg.zscore_std is not None:
            mean = torch.as_tensor(self.cfg.zscore_mean, dtype=torch.float32)
            std = torch.as_tensor(self.cfg.zscore_std, dtype=torch.float32)
            if mean.shape != std.shape:
                raise ValueError("zscore_mean and zscore_std must have the same shape.")
            if mean.numel() != input_dim:
                raise ValueError(
                    f"Z-score arrays must match concatenated input dimension ({input_dim}), "
                    f"got {mean.numel()}."
                )
            self._zscore_mean = mean
            self._zscore_std = std

        self._built = True
        if self.cfg.verbose:
            print(
                f"[ProprioTokenizer] Built tokenizer with input_dim={input_dim}, "
                f"hidden_dims={self.cfg.hidden_dims}, d_model={self.cfg.d_model}"
            )

    def forward(self, obs: Union[Dict[str, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
        if isinstance(obs, dict):
            x = self._concatenate_obs(obs)
        elif torch.is_tensor(obs):
            x = obs
        else:
            raise TypeError("Observations must be a dict or Tensor.")

        if x.ndim != 2:
            raise ValueError(f"Expected observation tensor of shape [B, dim], got {tuple(x.shape)}.")

        self._maybe_build(x.shape[-1])
        assert self.frontend is not None and self.token_head is not None

        x = self._normalize(x)
        x = self.frontend(x)
        if self.projector is not None:
            x = self.projector(x)
        tokens, pooled = self.token_head(x)
        return tokens, pooled

    def load_pretrained(self, ckpt: Union[str, Dict[str, Tensor]]) -> Tuple[bool, int]:
        if not self._built or self.frontend is None:
            raise RuntimeError("Tokenizer front-end not built yet. Run a forward pass first.")
        success, layers = self.frontend.load_actor_front_from_teacher(ckpt, verbose=self.cfg.verbose)
        if not success and self.cfg.require_strict_pretrained:
            raise RuntimeError("No pretrained actor layers were copied and require_strict_pretrained=True.")
        return success, layers


def _build_dummy_obs(batch: int = 2) -> Dict[str, Tensor]:
    tensors: Dict[str, Tensor] = {
        "base_lin_vel": torch.randn(batch, 3),
        "base_ang_vel": torch.randn(batch, 3),
        "projected_gravity": torch.randn(batch, 3),
        "joint_pos": torch.randn(batch, 12),
        "joint_vel": torch.randn(batch, 12),
        "actions": torch.randn(batch, 12),
    }
    return tensors


if __name__ == "__main__":
    cfg = ProprioTokenizerCfg()
    tokenizer = ProprioTokenizer(cfg)
    obs = _build_dummy_obs(batch=2)
    tokens, pooled = tokenizer(obs)

    teacher_ckpt = os.environ.get("TEACHER_CKPT")
    if teacher_ckpt:
        try:
            success, layers = tokenizer.load_pretrained(teacher_ckpt)
            print(f"copied {layers} Linear layers (success={success}) from {teacher_ckpt}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to load pretrained front-end: {exc}")

    print(f"tokens shape: {tuple(tokens.shape)}, pooled shape: {tuple(pooled.shape)}")
