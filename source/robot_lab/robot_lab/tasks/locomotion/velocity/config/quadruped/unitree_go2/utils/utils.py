from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence

import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from . import utils_cfg

from isaaclab.sensors import RayCaster, RayCasterCfg


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


class RayCasterVertical(RayCaster):
    """A RayCaster that uses a vertical grid pattern for ray casting in YZ plane."""
    
    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg=cfg)

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
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]