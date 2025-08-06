"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane, make_box
from .utils import make_overhangs_box

if TYPE_CHECKING:
    from . import track_terrains_cfg

def track_terrain(
    difficulty: float, cfg: track_terrains_cfg.TrackTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a track terrain for racing from one end to another.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # 创建跑道地形，机器人从一端跑到另一端
    # 地形尺寸: width=6m, length=12m
    # 机器人出生点在起始端 (y=0附近)，目标是到达终点 (y=12附近)
    
    # 设置原点在跑道起始端中央附近
    origin = (cfg.size[0] / 2.0, 1.0, 0.0)  # (3.0, 1.0, 0.0)

    meshes_list = list()
    
    # 创建平坦的跑道
    track_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    meshes_list += track_mesh

    # 将跑道均匀分成8个部分
    track_width = cfg.size[0]  # 4.0m
    track_length = cfg.size[1]  # 16.0m
    section_length = track_length / 8  

    # 第一部分是出生平台，不添加障碍物
    # 后面7个部分每个添加一个障碍物
    for section_index in range(1, 8):  # 从第2个部分开始（索引1-7）
        # 计算当前部分的Y坐标范围
        section_start_y = section_index * section_length
        section_end_y = (section_index + 1) * section_length
        section_center_y = (section_start_y + section_end_y) / 2
        
        # TODO: 在这个部分添加障碍物
        airbox = make_overhangs_box(
             length=2.0,
             width=1.5,
             height=0.5,
             center=(track_width / 2, section_center_y, 1.0)  # 中心位置
        )
        meshes_list += airbox

    # 可以在这里添加跑道标记线或边界（可选）
    # 例如在跑道两侧添加边界
    if cfg.border_width > 0.0:
            # obtain a list of meshes for the border
            border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
            border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
            make_borders = make_border(cfg.size, border_inner_size, cfg.border_height, border_center)
            # add the border meshes to the list of meshes
            meshes_list += make_borders

    return meshes_list, np.array(origin)

