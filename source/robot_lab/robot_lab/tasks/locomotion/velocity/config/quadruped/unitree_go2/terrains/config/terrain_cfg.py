"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg


from robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.terrains.track_terrain_generator_cfg import TrackTerrainGeneratorCfg
import robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.terrains as track_terrain_gen


TRACK_TERRAIN_CFG = TrackTerrainGeneratorCfg(
    size=(4.0, 32.0),
    border_width=20.0,
    num_rows=10,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # 使用自定义的跑道地形，三种地形等分
        # "track_overhang": track_terrain_gen.TrackTerrainOverhangCfg(
        #     proportion=0.33,
        #     border_width=0.5,
        #     border_height=0.2,
        # ),
        # "track_crack": track_terrain_gen.TrackTerrainCrackCfg(
        #     proportion=0.33,
        #     border_width=0.5,
        #     border_height=0.2,
        # ),
        "track_highland": track_terrain_gen.TrackTerrainHighlandCfg(
            proportion=1.0,
            border_width=0.5,
            border_height=0.2,
        ),

    },
)
"""Track terrain configuration for racing from one end to another."""

POST_DISASTER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # Mesh terrains
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.05,
        # ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.05,
        #     grid_width=0.45,
        #     grid_height_range=(0.05, 0.2),
        #     platform_width=2.0,
        # ),
        # "rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=0.05,
        #     rail_thickness_range=(0.1, 0.2),
        #     rail_height_range=(0.1, 0.3),
        #     platform_width=2.0,
        # ),
        # "pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.05,
        #     pit_depth_range=(0.2, 0.5),
        #     platform_width=2.0,
        #     double_pit=False,
        # ),
        # "box": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.05,
        #     box_height_range=(0.1, 0.3),
        #     platform_width=2.0,
        #     double_box=False,
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.05,
        #     gap_width_range=(0.2, 0.5),
        #     platform_width=2.0,
        # ),
        # "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
        #     proportion=0.05,
        #     ring_width_range=(0.3, 0.6),
        #     ring_height_range=(0.1, 0.3),
        #     ring_thickness=0.1,
        #     platform_width=2.0,
        # ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.05,
        #     num_bars=4,
        #     bar_width_range=(0.2, 0.4),
        #     bar_height_range=(0.1, 0.3),
        #     platform_width=2.0,
        # ),
        # # Height field terrains
        # "hf_random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.05,
        #     noise_range=(0.02, 0.10),
        #     noise_step=0.02,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.05,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.05,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
        #     proportion=0.05,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
        #     proportion=0.05,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.05,
        #     obstacle_height_mode="choice",
        #     obstacle_width_range=(0.2, 0.5),
        #     obstacle_height_range=(0.05, 0.2),
        #     num_obstacles=10,
        #     platform_width=2.0,
        # ),
        # "hf_wave": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.05,
        #     amplitude_range=(0.05, 0.15),
        #     num_waves=2,
        # ),
        # "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #     proportion=0.05,
        #     stone_height_max=0.2,
        #     stone_width_range=(0.3, 0.6),
        #     stone_distance_range=(0.2, 0.5),
        #     holes_depth=-10.0,
        #     platform_width=2.0,
        # ),
        # # Repeated objects terrains
        # "repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        #     proportion=0.05,
        #     object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        #         num_objects=5,
        #         height=0.1,
        #         radius=0.2,
        #         max_yx_angle=0.0,
        #         degrees=True,
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        #         num_objects=15,
        #         height=0.3,
        #         radius=0.4,
        #         max_yx_angle=15.0,
        #         degrees=True,
        #     ),
        #     max_height_noise=0.05,
        #     platform_width=2.0,
        # ),
        # "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.05,
        #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=5,
        #         height=0.1,
        #         size=(0.2, 0.2),
        #         max_yx_angle=0.0,
        #         degrees=True,
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=15,
        #         height=0.3,
        #         size=(0.4, 0.4),
        #         max_yx_angle=15.0,
        #         degrees=True,
        #     ),
        #     max_height_noise=0.05,
        #     platform_width=2.0,
        # ),
        # "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
        #     proportion=0.1,
        #     object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        #         num_objects=5,
        #         height=0.1,
        #         radius=0.2,
        #         max_yx_angle=0.0,
        #         degrees=True,
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        #         num_objects=15,
        #         height=0.3,
        #         radius=0.4,
        #         max_yx_angle=15.0,
        #         degrees=True,
        #     ),
        #     max_height_noise=0.05,
        #     platform_width=2.0,
        # ),
    },
)
"""All available terrains configuration with equal probability (0.05 each)."""
