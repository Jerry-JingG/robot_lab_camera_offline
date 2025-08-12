from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg

from .track_terrain_generator import TrackTerrainGenerator


@configclass
class TrackTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = TrackTerrainGenerator