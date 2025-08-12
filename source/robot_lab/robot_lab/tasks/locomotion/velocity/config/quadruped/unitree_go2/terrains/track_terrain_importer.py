from __future__ import annotations

from isaaclab.terrains import TerrainGenerator
from isaaclab.terrains import TerrainImporter
from typing import TYPE_CHECKING
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from .track_terrain_importer_cfg import TrackTerrainImporterCfg


from .track_terrain_generator import TrackTerrainGenerator


class TrackTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TrackTerrainImporterCfg):
        # print(cfg.terrain_generator)
        super().__init__(cfg)
        self.terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
        # 不新增额外缓存，按需动态生成

    @property
    def env_grid_indices(self):
        """返回当前每个 env 对应的 (row, col) 索引 (tensor[num_envs,2]).

        row 由 terrain_levels 给出 (难度行)
        col 由 terrain_types  给出 (子地形列)
        这样在更新 curriculum (update_env_origins) 后自动生效。
        若属性不存在则返回 None，保持兼容。
        """
        if hasattr(self, "terrain_levels") and hasattr(self, "terrain_types"):
            try:
                import torch  # 延迟导入，避免在无 torch 场景提前报错
                return torch.stack([self.terrain_levels.to(torch.long), self.terrain_types.to(torch.long)], dim=1)
            except Exception:
                return None
        return None
