# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
import robot_lab.tasks.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GoalVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) towards next terrain waypoint.

    若获取 waypoint 失败则退回父类均匀采样并做幅值阈值裁剪。
    仅最小侵入式修改: 复用原类, 不新增新类型。"""

    cfg: mdp.GoalVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.GoalVelocityCommandCfg, env: 'ManagerBasedEnv'):
        # 复用父类初始化
        super().__init__(cfg, env)
        # goal position command 开关（向后兼容: 若配置里没有该字段则默认为 False）
        self._use_goal_pos = getattr(self.cfg, "goal_position_command", False)
        if self._use_goal_pos:
            # 这里假设使用世界系 (x, y, z) 目标点；若只用平面，可忽略 z。
            self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            # 提供一个占位，避免属性不存在
            self.goal_pos_w = None
        # 记录期望航向（用于 debug 红色箭头）
        self.desired_heading = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
         # 原有的指令重采样逻辑 ...
        terrain = self._env.scene.terrain  # TerrainImporter

        # 1. 获取所有环境的张量（长度 = num_envs）
        terrain_levels = terrain.terrain_levels          # shape: [num_envs]
        terrain_types  = terrain.terrain_types           # shape: [num_envs]
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # goal position target（与 heading 同级，dummy 采样逻辑）
        if self._use_goal_pos:
            # Dummy: 在原点附近给一个固定或简单偏移；实际应在此采样/查询全局 waypoint。
            # 这里简单设置为 (1.0, 0.0, 0.0)，可改为随机：torch.randn_like -> 再缩放。
            # self.goal_pos_w[env_ids] = torch.tensor([-2.0, -7.0, 0.0], device=self.device)
            for i in env_ids:
                level=terrain_levels[i]
                types=terrain_types[i]
                self.goal_pos_w[i] = torch.as_tensor(terrain.terrain_generator.terrain_goals[level,types], device=self.device, dtype=self.goal_pos_w.dtype)
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # print(self.goal_pos_w)
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Compute angular velocity from goal position (dummy 实现)
        if self._use_goal_pos and self.goal_pos_w is not None:
            # 这里假设我们需要所有 env 指向各自的 goal；实际可加入 mask。
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            # 当前世界位置
            robot_pos_w = self.robot.data.root_pos_w
            # print(f"Robot positions: {robot_pos_w[env_ids, :2]}")  # 如需调试可开启
            # print(f"Goal positions: {self.goal_pos_w[env_ids, :2]}")  # 如需调试可开启
            # 方向向量（平面）
            diff = self.goal_pos_w[env_ids, :2] - robot_pos_w[env_ids, :2]
            desired_heading = torch.atan2(diff[:, 1], diff[:, 0])
            # 缓存期望航向（覆盖 heading_command 的值，若两者同时启用）
            self.desired_heading[env_ids] = desired_heading
            # 当前朝向
            current_heading = self.robot.data.heading_w[env_ids]
            heading_error = math_utils.wrap_to_pi(desired_heading - current_heading)
            ang_vel_cmd = self.cfg.heading_control_stiffness * heading_error
            # print(ang_vel_cmd)  # 如需调试可开启
            self.vel_command_b[env_ids, 2] = torch.clip(
                ang_vel_cmd,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
 
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            if not hasattr(self, "expected_heading_visualizer"):
                self.expected_heading_visualizer = VisualizationMarkers(self.cfg.expected_heading_visualizer_cfg)
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.expected_heading_visualizer.set_visibility(True)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
            if hasattr(self, "expected_heading_visualizer"):
                self.expected_heading_visualizer.set_visibility(False)
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        # 红色箭头：期望航向，长度固定为1（可按需调整）
        if hasattr(self, "expected_heading_visualizer"):
            desired_heading = self.desired_heading
            goal_pos_w = self.goal_pos_w.clone()
            goal_pos_w[:, 2] += 0.5  # 稍微抬高
            goal_pos_scale_factor = 20  # 可视化缩放因子 (0.2m直径)，您可以按需调整
            goal_pos_scales = torch.full((self.num_envs, 3), goal_pos_scale_factor, device=self.device)
            # print(f"Desired heading: {desired_heading}")
            # print(f"Goal positions: {goal_pos_w}")
            heading_scale, heading_quat = self._resolve_heading_to_arrow(desired_heading, torch.ones_like(desired_heading))
            self.expected_heading_visualizer.visualize(base_pos_w, heading_quat, heading_scale)
            self.goal_pos_visualizer.visualize(goal_pos_w, marker_indices=[0], scales=goal_pos_scales)

    def _resolve_heading_to_arrow(self, heading_angle: torch.Tensor, length: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """将航向角(世界系绕Z的 yaw) 转换成箭头缩放与姿态四元数。

        NOTE:
            期望航向箭头需要 *始终指向世界系目标点*，不随机器人自旋转，所以这里返回的四元数直接是世界系 yaw 旋转，**不再与 base 朝向相乘**。
            如果未来需要相对机体朝向的显示，可再新增参数控制是否乘以 base_quat_w。

        Args:
            heading_angle: (N,) 航向角，以弧度表示 (世界系 yaw)。
            length: (N,) 可选长度系数；若为 None 则使用 1。最终应用到 x 方向尺度 (箭头指向 x)。
        Returns:
            (arrow_scale, arrow_quat)
        """
        if length is None:
            length = torch.ones_like(heading_angle)
        # 基础缩放（取绿色可视化的默认 scale，保持三个箭头风格一致）
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(heading_angle.shape[0], 1)
        # x 方向长度按 length * 3.0（与之前速度箭头一致放大因子）
        arrow_scale[:, 0] *= length * 3.0
        zeros = torch.zeros_like(heading_angle)
        # 直接构造世界系绕 Z 的四元数 (roll=0,pitch=0,yaw=heading)，不再乘 base 姿态
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat
        
@configclass
class GoalVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the goal velocity command generator."""

    class_type: type = GoalVelocityCommand
    # 新增开关（默认关闭，避免影响现有实验）
    goal_position_command: bool = False
    goal_pos_visualizer_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_point",
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    expected_heading_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_expected"
    )
    """红色箭头: 期望航向方向。"""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    expected_heading_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
