from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
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