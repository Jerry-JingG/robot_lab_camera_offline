"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane, make_box
from .utils import make_overhangs_box, make_narrow_crack, make_highland_platform

if TYPE_CHECKING:
    from . import track_terrains_cfg

def track_terrain_overhang(
    difficulty: float, cfg: track_terrains_cfg.TrackTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, list[np.ndarray]]:
    """Generate a track terrain for racing from one end to another.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain, the origin of the terrain (in m), and waypoints.
    """
    # 创建跑道地形，机器人从一端跑到另一端
    # 地形尺寸: width=6m, length=12m
    # 机器人出生点在起始端 (y=0附近)，目标是到达终点 (y=12附近)
    
    # 设置原点在跑道起始端中央附近
    origin = (cfg.size[0] / 2.0, 1.0, 0.0)  # (3.0, 1.0, 0.0)

    meshes_list = list()
    waypoints = list()
    
    # 创建平坦的跑道
    track_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    meshes_list += track_mesh

    # 将跑道均匀分成8个部分
    track_width = cfg.size[0]  # 4.0m
    track_length = cfg.size[1]  # 32.0m
    section_length = track_length / 8  

    # 第一部分是出生平台，不添加障碍物
    # 后面6个部分每个添加一个障碍物
    for section_index in range(1, 7):  # 从第2个部分开始（索引1-6）
        # 计算当前部分的Y坐标范围
        section_start_y = section_index * section_length
        section_end_y = (section_index + 1) * section_length
        section_center_y = (section_start_y + section_end_y) / 2
        
        # 基于 difficulty 生成悬空障碍 (难度 0 简单, 1 最难)
        # 尺寸随难度增大: 更长/更宽/更厚, 同时降低离地间隙增加通过难度
        overhang_length = 1.5 + 1.0 * difficulty      # 1.5 ~ 2.5
        overhang_width  = 1.0     # 1.0 ~ 1.8
        overhang_height = 0.4      # 0.4 ~ 0.8 (障碍本体厚度)
        clearance_low   = 0.25                        # 高难度时底面高度
        clearance_high  = 0.4                         # 低难度时底面高度
        bottom_height   = clearance_high - (clearance_high - clearance_low) * difficulty
        center_z        = bottom_height + overhang_height / 2.0
        airbox = make_overhangs_box(
            length=overhang_length,
            width=overhang_width,
            height=overhang_height,
            center=(track_width / 2, section_center_y, center_z),
        )
        meshes_list += airbox
        
        # 在障碍物中心位置的地面上生成waypoint
        waypoint = np.array([track_width / 2, section_center_y, 0.0])
        waypoints.append(waypoint)
    waypoints.append(np.array([track_width / 2, track_length - 1.0, 0.0]))  # 最终目标点
    # print("+++++++++++++++++++++++++++++++++")
    # print(waypoints[-1])
    # 可以在这里添加跑道标记线或边界（可选）
    # 例如在跑道两侧添加边界
    if cfg.border_width > 0.0:
            # obtain a list of meshes for the border
            border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
            border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
            make_borders = make_border(cfg.size, border_inner_size, cfg.border_height, border_center)
            # add the border meshes to the list of meshes
            meshes_list += make_borders

    return meshes_list, np.array(origin), waypoints


def track_terrain_crack(
    difficulty: float, cfg: track_terrains_cfg.TrackTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, list[np.ndarray]]:
    """Generate a track terrain for racing from one end to another.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain, the origin of the terrain (in m), and waypoints.
    """
    # 创建跑道地形，机器人从一端跑到另一端
    # 地形尺寸: width=6m, length=12m
    # 机器人出生点在起始端 (y=0附近)，目标是到达终点 (y=12附近)
    
    # 设置原点在跑道起始端中央附近
    origin = (cfg.size[0] / 2.0, 1.0, 0.0)  # (3.0, 1.0, 0.0)

    meshes_list = list()
    waypoints = list()
    
    # 创建平坦的跑道
    track_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    meshes_list += track_mesh

    # 将跑道均匀分成8个部分
    track_width = cfg.size[0]  # 4.0m
    track_length = cfg.size[1]  # 16.0m
    section_length = track_length / 8  

    # 第一部分是出生平台，不添加障碍物
    # 后面6个部分每个添加一个障碍物
    for section_index in range(1, 7):  # 从第2个部分开始（索引1-6）
        # 计算当前部分的Y坐标范围
        section_start_y = section_index * section_length
        section_end_y = (section_index + 1) * section_length
        section_center_y = (section_start_y + section_end_y) / 2
        
        # 基于 difficulty 生成窄缝障碍
        # 难度越高: 缝更窄 / 障碍更深更高
        crack_total_length = track_width*0.75                       # 缝的总长度等于跑道宽度，顶到两边边界
        crack_depth        = 0.8 + 0.9 * difficulty            # 0.8 ~ 1.7 (沿 y)
        crack_height       = 1.0                  
        gap_min_width      = 0.3
        gap_base_width     = 0.5
        gap_width          = max(gap_min_width, gap_base_width - 0.6 * difficulty)  # 0.5 ~ 0.3
        center_z           = crack_height / 2.0
        narrow_gap = make_narrow_crack(
            length=crack_total_length,
            width=crack_depth,
            height=crack_height,
            center=(track_width / 2, section_center_y, center_z),
            gap_x_offset=0.0,
            gap_width=gap_width,
        )
        meshes_list += narrow_gap

        # 在障碍物中心位置的地面上生成waypoint
        waypoint = np.array([track_width / 2, section_center_y, 0.0])
        waypoints.append(waypoint)
    waypoints.append(np.array([track_width / 2, track_length - 1.0, 0.0]))  # 最终目标点

    # 可以在这里添加跑道标记线或边界（可选）
    # 例如在跑道两侧添加边界
    if cfg.border_width > 0.0:
            # obtain a list of meshes for the border
            border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
            border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
            make_borders = make_border(cfg.size, border_inner_size, cfg.border_height, border_center)
            # add the border meshes to the list of meshes
            meshes_list += make_borders

    return meshes_list, np.array(origin), waypoints


def track_terrain_highland(
    difficulty: float, cfg: track_terrains_cfg.TrackTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, list[np.ndarray]]:
    """Generate a track terrain with highland platforms for racing from one end to another.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain, the origin of the terrain (in m), and waypoints.
    """
    # 创建跑道地形，机器人从一端跑到另一端
    # 地形尺寸: width=6m, length=12m
    # 机器人出生点在起始端 (y=0附近)，目标是到达终点 (y=12附近)
    
    # 设置原点在跑道起始端中央附近
    origin = (cfg.size[0] / 2.0, 1.0, 0.0)  # (3.0, 1.0, 0.0)

    meshes_list = list()
    waypoints = list()
    
    # 创建平坦的跑道
    track_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    meshes_list += track_mesh

    # 将跑道均匀分成8个部分
    track_width = cfg.size[0]  # 4.0m
    track_length = cfg.size[1]  # 16.0m
    section_length = track_length / 8  

    # 第一部分是出生平台，不添加障碍物
    # 后面6个部分每个添加一个障碍物
    for section_index in range(1, 7):  # 从第2个部分开始（索引1-6）
        # 计算当前部分的Y坐标范围
        section_start_y = section_index * section_length
        section_end_y = (section_index + 1) * section_length
        section_center_y = (section_start_y + section_end_y) / 2
        
        # 基于 difficulty 生成高台阶障碍
        # 难度越高: 台阶更高
        platform_length = track_width * 0.75                    # 平台长度
        platform_width = 2.0                                    # 平台宽度 (沿 y)
        platform_height = 0.05 + 0.5 * difficulty              # 0.05 ~ 0.55
        center_z = 0.0  # 地面高度作为参考
        
        highland = make_highland_platform(
            length=platform_length,
            width=platform_width,
            height=platform_height,
            center=(track_width / 2, section_center_y, center_z),
        )
        meshes_list += highland

        # 在障碍物中心位置的地面上生成waypoint
        waypoint = np.array([track_width / 2, section_center_y, 0.0])
        waypoints.append(waypoint)
    
    waypoints.append(np.array([track_width / 2, track_length - 1.0, 0.0]))  # 最终目标点

    # 可以在这里添加跑道标记线或边界（可选）
    # 例如在跑道两侧添加边界
    if cfg.border_width > 0.0:
            # obtain a list of meshes for the border
            border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
            border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
            make_borders = make_border(cfg.size, border_inner_size, cfg.border_height, border_center)
            # add the border meshes to the list of meshes
            meshes_list += make_borders

    return meshes_list, np.array(origin), waypoints

