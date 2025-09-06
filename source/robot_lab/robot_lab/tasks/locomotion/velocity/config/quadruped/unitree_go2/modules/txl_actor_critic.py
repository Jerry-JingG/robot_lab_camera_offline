"""
TXLActorCritic: A drop-in Actor-Critic module for rsl_rl PPO that replaces the
default MLP with:
  - ProprioTokenizer -> [B, C]
  - VisualTokenizer  -> [B, N, C]
  - TXLBackbone with memory across steps -> [B, 1+N, C]
  - PolicyHeads over pooled tokens -> actor mean [B, A], critic value [B, 1]

This class matches the API expected by rsl_rl.algorithms.PPO:
  - act(observations) -> sample actions; exposes distribution, action_mean/std
  - get_actions_log_prob(actions)
  - act_inference(observations) -> deterministic actions (mean)
  - evaluate(critic_observations) -> value tensor [B,1]
  - reset(dones) -> clears memory for finished envs

Assumptions:
  - Actor observations are a flat tensor [B, num_actor_obs] that concatenates
    [proprio (proprio_dim), depth (C*H*W)].
  - Critic observations follow the same layout unless configured otherwise.

If your critic obs differ (e.g., privileged features), set critic_proprio_dim
and critic_depth_shape accordingly, or set use_critic_vision=False to ignore
vision for critic and run critic on proprio only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .visual_tokenizer import VisualTokenizer
from .proprio_tokenizer import ProprioTokenizer
from .txl_backbone import TXLBackbone, PolicyHeads
from rsl_rl.modules.actor_critic import ActorCritic as ACBase


@dataclass
class TXLActorCriticCfg:
    # Observation parsing for actor
    proprio_dim: int = 45
    depth_shape: Tuple[int, int, int] = (4, 64, 64)  # (C,H,W)
    # Observation parsing for critic (defaults to actor layout)
    critic_proprio_dim: Optional[int] = None
    critic_depth_shape: Optional[Tuple[int, int, int]] = None
    use_critic_vision: bool = True

    # Token/Backbone dimensions
    token_dim: int = 128
    visual_out_tokens: int = 16  # VisualTokenizer for 64x64 -> 4x4 = 16

    # TXL hyper-parameters
    nhead: int = 4
    num_layers: int = 4
    ffn_hidden: int = 512
    dropout: float = 0.0
    mem_len_frames: int = 128

    # Action space
    action_dim: int = 12


class TXLActorCritic(ACBase):
    is_recurrent = False  # We keep PPO non-recurrent; memory is used only online.

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        activation: str | None = None,  # ignored; kept for compatibility
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # TXL-specific kwargs (optional; safe defaults)
        proprio_dim: int = 45,
        depth_shape: Tuple[int, int, int] = (4, 64, 64),
        critic_proprio_dim: Optional[int] = None,
        critic_depth_shape: Optional[Tuple[int, int, int]] = None,
        use_critic_vision: bool = True,
        token_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_hidden: int = 512,
        dropout: float = 0.0,
        mem_len_frames: int = 128,
        visual_out_tokens: int = 16,
        **kwargs,
    ) -> None:
        # Do not call ACBase.__init__ (it builds an MLP we don't need)
        nn.Module.__init__(self)

        # Save parsing / model cfg
        self.cfg = TXLActorCriticCfg(
            proprio_dim=proprio_dim,
            depth_shape=depth_shape,
            critic_proprio_dim=critic_proprio_dim,
            critic_depth_shape=critic_depth_shape,
            use_critic_vision=use_critic_vision,
            token_dim=token_dim,
            visual_out_tokens=visual_out_tokens,
            nhead=nhead,
            num_layers=num_layers,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            mem_len_frames=mem_len_frames,
            action_dim=num_actions,
        )

        # Validate shapes vs provided obs dims (allow extra non-tokenized terms)
        c, h, w = self.cfg.depth_shape
        depth_elems = c * h * w
        min_actor = self.cfg.proprio_dim + depth_elems
        if num_actor_obs < min_actor:
            raise ValueError(
                f"Actor obs dim too small: need at least {min_actor} (proprio {self.cfg.proprio_dim} + depth {depth_elems}), got {num_actor_obs}"
            )

        if self.cfg.critic_proprio_dim is None:
            self.cfg.critic_proprio_dim = self.cfg.proprio_dim
        if self.cfg.critic_depth_shape is None:
            self.cfg.critic_depth_shape = self.cfg.depth_shape

        cp = self.cfg.critic_proprio_dim if self.cfg.critic_proprio_dim is not None else self.cfg.proprio_dim
        if self.cfg.use_critic_vision:
            cc, ch, cw = self.cfg.critic_depth_shape
            min_critic = cp + cc * ch * cw
        else:
            min_critic = cp
        if num_critic_obs < min_critic:
            raise ValueError(
                f"Critic obs dim too small: need at least {min_critic}, got {num_critic_obs}. Check critic_* dims or set use_critic_vision=False."
            )

        # Modules
        self.prop_tok = ProprioTokenizer(in_dim=self.cfg.proprio_dim, token_dim=self.cfg.token_dim)
        self.vis_tok = VisualTokenizer()  # fixed to (4,64,64)-> [B,16,128]
        self.backbone = TXLBackbone(
            d_model=self.cfg.token_dim,
            nhead=self.cfg.nhead,
            num_layers=self.cfg.num_layers,
            ffn_hidden=self.cfg.ffn_hidden,
            dropout=self.cfg.dropout,
            mem_len_frames=self.cfg.mem_len_frames,
            tokens_per_frame=1 + self.cfg.visual_out_tokens,
        )
        self.heads = PolicyHeads(c=self.cfg.token_dim, action_dim=num_actions)

        # Action noise params
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        # Memory: per-layer memory tensor or None; initialized lazily on first act()
        self.mems = None

        # Distribution placeholder
        self.distribution = None

    # -----------------
    # Helper: parsing
    # -----------------
    def _split_actor_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """Split policy group obs per provided layout.

                Policy terms (by index):
                    0..5 -> proprio (3+3+3+12+12+12 = 45)
                    6..7 -> non-tokenized (height_scan 187, collision_scan 30)
                    8    -> front_cam_depth (16384)
                We extract proprio from the first 45 dims and depth from the last 16384 dims.
                """
                B = obs.shape[0]
                p = self.cfg.proprio_dim  # 45
                c, h, w = self.cfg.depth_shape
                depth_elems = c * h * w   # 16384
                if obs.shape[1] < p + depth_elems:
                        raise AssertionError(f"Policy obs too small: {obs.shape[1]} < {p + depth_elems}")
                prop = obs[:, :p]
                depth_flat = obs[:, -depth_elems:]
                depth = depth_flat.view(B, c, h, w)
                return prop, depth

    def _split_critic_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split critic group obs per provided layout.

        Critic terms (by index):
          0           -> base_lin_vel (3) [ignored for proprio token]
          1..6        -> proprio used by tokenizer (same 45 dims as policy)
          7..8        -> non-tokenized (height_scan 187, collision_scan 30)
          9           -> front_cam_depth (16384)
        We extract proprio from indices covering 1..6 (total 45 dims) and depth from last 16384 dims.
        If use_critic_vision=False, we return None for depth.
        """
        B = obs.shape[0]
        cp = self.cfg.critic_proprio_dim if self.cfg.critic_proprio_dim is not None else self.cfg.proprio_dim
        if not self.cfg.use_critic_vision:
            # Take only the 45 proprio dims starting after the first 3 (base_lin_vel)
            prop = obs[:, 3 : 3 + cp]
            return prop, None
        c, h, w = self.cfg.critic_depth_shape
        depth_elems = c * h * w
        if obs.shape[1] < 3 + cp + depth_elems:
            raise AssertionError(
                f"Critic obs too small: {obs.shape[1]} < {3 + cp + depth_elems}"
            )
        # Skip first 3 dims (base_lin_vel), then take 45 dims
        prop = obs[:, 3 : 3 + cp]
        depth_flat = obs[:, -depth_elems:]
        depth = depth_flat.view(B, c, h, w)
        return prop, depth

    # -----------------
    # API expected by PPO
    # -----------------
    def update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    @property
    def action_mean(self):
        return self.distribution.mean  # type: ignore

    @property
    def action_std(self):
        return self.distribution.stddev  # type: ignore

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)  # type: ignore

    def _forward_tokens(self, prop: torch.Tensor, depth: torch.Tensor, use_mem: bool) -> torch.Tensor:
        # Disable memory during gradient-enabled contexts (e.g., PPO updates)
        # to avoid shape mismatches with self.mems when batches are shuffled.
        if use_mem and torch.is_grad_enabled():
            use_mem = False
        t_prop = self.prop_tok(prop)           # [B, C]
        t_vis = self.vis_tok(depth)            # [B, N, C]
        if use_mem:
            if self.mems is None:
                # initialize as explicit Optional[Tensor] elements
                self.mems = [None for _ in range(self.backbone.num_layers)]
            x_seq, self.mems = self.backbone(t_prop, t_vis, mems=self.mems, attn_mask=None)
        else:
            # No memory during updates (batches are shuffled)
            x_seq, _ = self.backbone(
                t_prop,
                t_vis,
                mems=[None for _ in range(self.backbone.num_layers)],
                attn_mask=None,
            )
        return x_seq

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        prop, depth = self._split_actor_obs(observations)
        x_seq = self._forward_tokens(prop, depth, use_mem=True)
        mean, value = self.heads(x_seq)
        # store latest value for potential use (not required by PPO API here)
        self._last_value = value
        self.update_distribution(mean)
        return self.distribution.sample()  # type: ignore

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        prop, depth = self._split_actor_obs(observations)
        x_seq = self._forward_tokens(prop, depth, use_mem=True)
        mean, _ = self.heads(x_seq)
        return mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        prop, depth = self._split_critic_obs(critic_observations)
        if depth is None:
            # Build dummy visual tokens as zeros if vision disabled for critic
            B = prop.shape[0]
            t_prop = self.prop_tok(prop)
            t_vis = torch.zeros(B, self.cfg.visual_out_tokens, self.cfg.token_dim, device=prop.device, dtype=prop.dtype)
            x_seq, _ = self.backbone(t_prop, t_vis, mems=[None] * self.backbone.num_layers, attn_mask=None)
        else:
            x_seq = self._forward_tokens(prop, depth, use_mem=False)
        _, value = self.heads(x_seq)
        return value

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        if self.mems is None:
            return
        if dones is None:
            return
        if dones.ndim == 2:
            done_mask = dones[:, 0] > 0
        else:
            done_mask = dones > 0
        if not torch.any(done_mask):
            return
        # Zero out memories for finished envs (keeps tensor shapes intact)
        for i, mem in enumerate(self.mems):
            if mem is None:
                continue
            mem[done_mask] = 0
