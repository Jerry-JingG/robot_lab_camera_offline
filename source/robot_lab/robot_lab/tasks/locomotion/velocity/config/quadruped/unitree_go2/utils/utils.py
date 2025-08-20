from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
from dataclasses import dataclass

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from . import utils_cfg

from isaaclab.sensors import RayCaster, RayCasterCfg, RayCasterData


def grid_pattern_vertical(cfg: utils_cfg.GridPatternVerticalCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 从isaaclab.sensors.patterns复制一个pattern函数并改写成vertical形式（YZ平面）
    """A regular grid pattern for ray casting in YZ plane.

    The grid pattern is made from rays that are parallel to each other. They span a 2D grid in the sensor's
    local YZ coordinates from ``(-width/2, -height/2)`` to ``(width/2, height/2)``, which is defined
    by the ``size = (width, height)`` and ``resolution`` parameters in the config.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.

    Raises:
        ValueError: If the ordering is not "xy" or "yx".
        ValueError: If the resolution is less than or equal to 0.
    """
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
    if cfg.resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")

    # resolve mesh grid indexing (note: torch meshgrid is different from numpy meshgrid)
    # check: https://github.com/pytorch/pytorch/issues/15301
    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    # define grid pattern
    x = torch.arange(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    y = torch.arange(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

    # store into ray starts (YZ plane instead of XY plane)
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    # X坐标保持为0，表示在YZ平面上
    ray_starts[:, 1] = grid_x.flatten()  # Y坐标 (对应原来的X)
    ray_starts[:, 2] = grid_y.flatten()  # Z坐标 (对应原来的Y)

    # define ray-cast directions
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)

    return ray_starts, ray_directions


@dataclass
class RayCasterVerticalData(RayCasterData):
    """Data class for vertical ray caster sensor data."""
    # 继承自RayCasterData，添加必要的属性
    ray_starts_w: torch.Tensor = None


class RayCasterVertical(RayCaster):
    """A RayCaster that uses a vertical grid pattern for ray casting in YZ plane."""
    # Add a type hint for the instance variable `_data` to inform the static type checker.
    _data: RayCasterVerticalData

    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg=cfg)
        self._data = RayCasterVerticalData()

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._data.ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # check if user provided attach_yaw_only flag
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            # set ray alignment to yaw
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            # log the warning
            omni.log.warn(msg)
        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "world":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            # no rotation is considered and directions are not rotated
            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)
        # store the ray starts in world coordinates
        self._data.ray_starts_w[env_ids] = ray_starts_w


def calculate_euclidean_distance(point1: torch.Tensor, point2: torch.Tensor) -> torch.Tensor:
    """Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point tensor of shape (..., 3)
        point2: Second point tensor of shape (..., 3)
        
    Returns:
        Euclidean distance tensor of shape (...)
    """
    return torch.linalg.norm(point1 - point2, dim=-1)