from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


@dataclass
class VisionTokenizerCfg:
    """Configuration bundle for the depth visual tokenizer."""

    in_channels: int
    img_size: Tuple[int, int] = (128, 160)
    normalize: Dict[str, Any] = field(
        default_factory=lambda: {"min_depth": 0.0, "max_depth": 5.0}
    )
    d_model: int = 512
    num_tokens: int = 16
    attn_heads: int = 8
    patch_size: int = 16
    use_learnable_queries: bool = True
    freeze_frontend: bool = True
    require_strict_pretrained: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3, got {self.in_channels}.")

        min_depth = float(self.normalize.get("min_depth", 0.0))
        max_depth = float(self.normalize.get("max_depth", 0.0))
        if max_depth <= min_depth:
            raise ValueError(
                f"normalize.max_depth ({max_depth}) must be greater than "
                f"normalize.min_depth ({min_depth})."
            )
        self.normalize["min_depth"] = min_depth
        self.normalize["max_depth"] = max_depth

        if self.d_model % self.attn_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by attn_heads ({self.attn_heads})."
            )
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")


class _ConvNormAct(nn.Sequential):
    """Small helper block for the fallback frontend."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class VisualFrontEnd(nn.Module):
    """Visual CNN frontend with optional initialization from a teacher checkpoint."""

    KEYWORDS = ("vision", "cnn", "encoder", "backbone", "conv")

    def __init__(self, cfg: VisionTokenizerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.expected_in_channels = cfg.in_channels

        channels = [cfg.in_channels, 64, 128, 256]
        layers: list[nn.Module] = []
        for idx in range(1, len(channels)):
            stride = 2 if idx == 1 else 1  # Initial downsample keeps compute modest.
            layers.append(_ConvNormAct(channels[idx - 1], channels[idx], stride=stride))
        layers.extend(
            [
                nn.Conv2d(channels[-1], cfg.d_model, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(cfg.d_model),
                nn.GELU(),
            ]
        )
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def load_visual_front_from_teacher(
        self, teacher_state: Mapping[str, Tensor], verbose: bool = True
    ) -> Tuple[bool, int, int]:
        """Copy CNN parameters from a teacher checkpoint if shapes match."""

        def _find_key(
            shape: torch.Size, suffixes: Iterable[str], used: set[str]
        ) -> Optional[str]:
            for key, tensor in teacher_state.items():
                if key in used:
                    continue
                low = key.lower()
                if not any(token in low for token in self.KEYWORDS):
                    continue
                if not any(low.endswith(suf) for suf in suffixes):
                    continue
                if tensor.shape == shape:
                    return key
            return None

        conv_copied = 0
        norm_copied = 0
        used_keys: set[str] = set()
        logger = LOGGER if verbose else None

        with torch.no_grad():
            for name, module in self.named_modules():
                if not name:
                    continue
                if isinstance(module, nn.Conv2d):
                    weight_key = _find_key(module.weight.shape, ("weight",), used_keys)
                    if weight_key is None:
                        continue
                    module.weight.copy_(teacher_state[weight_key])
                    used_keys.add(weight_key)
                    if module.bias is not None:
                        bias_key = _find_key(module.bias.shape, ("bias",), used_keys)
                        if bias_key is not None:
                            module.bias.copy_(teacher_state[bias_key])
                            used_keys.add(bias_key)
                    self.expected_in_channels = module.weight.shape[1]
                    conv_copied += 1
                    if logger:
                        logger.info("Copied conv weights for %s from %s", name, weight_key)
                elif isinstance(
                    module,
                    (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm),
                ):
                    weight_key = _find_key(module.weight.shape, ("weight",), used_keys)
                    bias_key = (
                        _find_key(module.bias.shape, ("bias",), used_keys)
                        if module.bias is not None
                        else None
                    )
                    if weight_key is None:
                        continue
                    module.weight.copy_(teacher_state[weight_key])
                    used_keys.add(weight_key)
                    if bias_key is not None:
                        module.bias.copy_(teacher_state[bias_key])
                        used_keys.add(bias_key)
                    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        for attr, suf in (("running_mean", "running_mean"), ("running_var", "running_var")):
                            buf = getattr(module, attr)
                            match_key = _find_key(buf.shape, (suf,), used_keys)
                            if match_key is not None:
                                buf.copy_(teacher_state[match_key])
                                used_keys.add(match_key)
                    norm_copied += 1
                    if logger:
                        logger.info("Copied norm weights for %s", name)

        return conv_copied > 0, conv_copied, norm_copied


class VisionTokenizer(nn.Module):
    """Depth-aware visual tokenizer for Student fusion models."""

    def __init__(self, cfg: VisionTokenizerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = VisualFrontEnd(cfg)
        self.patch_embed = nn.Conv2d(
            cfg.d_model,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=False,
        )
        self.token_norm = nn.LayerNorm(cfg.d_model)

        if cfg.use_learnable_queries:
            self.query_tokens = nn.Parameter(torch.randn(cfg.num_tokens, cfg.d_model))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=cfg.d_model, num_heads=cfg.attn_heads, batch_first=True
            )
        else:
            self.register_parameter("query_tokens", None)
            self.cross_attn = None

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected rank-4 depth tensor [B,C,H,W], got {x.shape}.")

        norm_cfg = self.cfg.normalize
        min_depth = float(norm_cfg["min_depth"])
        max_depth = float(norm_cfg["max_depth"])
        x = torch.clamp(x, min=min_depth, max=max_depth)
        depth_range = max_depth - min_depth
        if depth_range <= 0:
            raise ValueError("normalize.max_depth must be greater than normalize.min_depth.")
        x = (x - min_depth) / depth_range

        if x.size(1) == 1 and self.frontend.expected_in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != self.frontend.expected_in_channels:
            raise ValueError(
                f"Input channels ({x.size(1)}) do not match frontend expectation "
                f"({self.frontend.expected_in_channels})."
            )

        mean_vals = norm_cfg.get("mean")
        std_vals = norm_cfg.get("std")
        if mean_vals is not None and std_vals is not None:
            mean = self._expand_stats(mean_vals, x, "mean")
            std = self._expand_stats(std_vals, x, "std").clamp_min(1e-6)
            x = (x - mean) / std

        features = self.frontend(x)
        patches = self.patch_embed(features)
        grid_tokens = patches.flatten(2).transpose(1, 2)  # [B, N, d_model]

        if self.cfg.use_learnable_queries and self.cross_attn is not None:
            queries = self.query_tokens.unsqueeze(0).expand(grid_tokens.size(0), -1, -1)
            tokens, _ = self.cross_attn(queries, grid_tokens, grid_tokens)
        else:
            tokens = self._pool_tokens(grid_tokens)

        return self.token_norm(tokens)

    def _pool_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.size(1) == self.cfg.num_tokens:
            return tokens
        pooled = F.adaptive_avg_pool1d(
            tokens.transpose(1, 2), output_size=self.cfg.num_tokens
        ).transpose(1, 2)
        return pooled

    def _expand_stats(self, values: Sequence[float], x: Tensor, name: str) -> Tensor:
        stats = torch.as_tensor(values, dtype=x.dtype, device=x.device)
        channels = x.size(1)
        if stats.numel() not in (1, channels):
            raise ValueError(
                f"normalize.{name} must have 1 or {channels} values, got {stats.numel()}."
            )
        if stats.numel() == 1 and channels > 1:
            stats = stats.expand(channels)
        return stats.view(1, channels, 1, 1)

    def load_pretrained(
        self, path_or_state: Union[str, Path, Mapping[str, Any]]
    ) -> Tuple[bool, int, int]:
        teacher_state = _resolve_state_dict(path_or_state)
        success, conv_copied, norm_copied = self.frontend.load_visual_front_from_teacher(
            teacher_state, verbose=self.cfg.verbose
        )

        if self.cfg.require_strict_pretrained and not success:
            raise RuntimeError(
                "require_strict_pretrained=True but no matching conv weights were copied."
            )

        if self.cfg.freeze_frontend:
            for param in self.frontend.parameters():
                param.requires_grad = False

        if self.cfg.verbose:
            LOGGER.info(
                "Pretrained front-end copy result: success=%s, convs=%d, norms=%d",
                success,
                conv_copied,
                norm_copied,
            )

        return success, conv_copied, norm_copied


def _resolve_state_dict(
    path_or_state: Union[str, Path, Mapping[str, Any]]
) -> Mapping[str, Tensor]:
    if isinstance(path_or_state, (str, Path)):
        path = Path(path_or_state)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        data = torch.load(path, map_location="cpu")
    else:
        data = path_or_state

    if isinstance(data, Mapping):
        tensor_like = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        if tensor_like:
            return tensor_like
        for key in ("state_dict", "model", "model_state_dict", "module"):
            candidate = data.get(key)
            if isinstance(candidate, Mapping):
                return _resolve_state_dict(candidate)

    raise ValueError("Unable to extract a state_dict from the provided checkpoint.")


def _dummy_depth(cfg: VisionTokenizerCfg) -> Tensor:
    return torch.rand(2, cfg.in_channels, cfg.img_size[0], cfg.img_size[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = VisionTokenizerCfg(in_channels=1)
    tokenizer = VisionTokenizer(cfg)

    teacher_path = os.environ.get("TEACHER_CKPT")
    copied = (False, 0, 0)
    if teacher_path:
        try:
            copied = tokenizer.load_pretrained(teacher_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load teacher checkpoint %s: %s", teacher_path, exc)

    depth = _dummy_depth(cfg)
    tokens = tokenizer(depth)

    print(f"Visual tokens shape: {tuple(tokens.shape)}")
    if teacher_path:
        success, conv_count, norm_count = copied
        print(
            f"Teacher copy -> success={success} "
            f"(conv={conv_count}, norm={norm_count}) from {teacher_path}"
        )
