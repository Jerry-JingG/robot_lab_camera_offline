from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, CameraCfg
import isaaclab.sim as sim_utils
from collections.abc import Callable
from . import utils


@configclass 
class GridPatternVerticalCfg(patterns.GridPatternCfg):
    """Vertical grid pattern configuration for ray casting in YZ plane."""
    # 继承自GridPatternCfg，重写direction默认值和func
    func: Callable = utils.grid_pattern_vertical
    direction: tuple = (1.0, 0.0, 0.0)  # 默认向前方（X轴正方向）发射


@configclass
class RayCasterVerticalCfg(RayCasterCfg):
    """RayCaster configuration using a vertical grid pattern."""
    # 继承自RayCasterCfg，重写pattern_cfg默认值
    class_type: type = utils.RayCasterVertical


# @configclass
# class FrontCameraCfg(CameraCfg):
#     """Front camera mounted on base for Unitree Go2."""
#     prim_path: str = "{ENV_REGEX_NS}/Robot/base/FrontCamera"
#     width: int = 64
#     height: int = 64
#     data_types: list[str] = ["distance_to_image_plane"]
#     # ROS convention: +Z forward, -Y up. Slight pitch down.
#     offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(pos=(0.25, 0.0, 0.20), rot=(-15.0, 0.0, 0.0, 0.0), convention="ros")
#     spawn: object = sim_utils.PinholeCameraCfg(
#         focal_length=16.0,
#         horizontal_aperture=20.955,
#         clipping_range=(0.1, 10.0),
#     )
#     update_period: float = 0.02
#     depth_clipping_behavior: str = "max"