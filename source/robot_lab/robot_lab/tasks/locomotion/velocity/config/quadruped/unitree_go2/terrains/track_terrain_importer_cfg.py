from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter

from .track_terrain_importer import TrackTerrainImporter

if TYPE_CHECKING:
    from .track_terrain_generator_cfg import TrackTerrainGeneratorCfg


@configclass
class TrackTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the track terrain importer."""

    class_type: type = TrackTerrainImporter
    """The class type of the terrain importer."""

    terrain_generator: TrackTerrainGeneratorCfg = MISSING