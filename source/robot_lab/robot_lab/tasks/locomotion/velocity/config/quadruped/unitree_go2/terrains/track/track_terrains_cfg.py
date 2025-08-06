from dataclasses import MISSING
from typing import Literal

import robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.terrains.track.track_terrains as track_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""


@configclass
class TrackTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a track terrain."""

    function = track_terrains.track_terrain

    border_width: float = 0.0

    border_height: float = 0.0

