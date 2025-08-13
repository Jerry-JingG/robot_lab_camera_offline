# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import rsl_rl_utils
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import robot_lab.tasks  # noqa: F401

try:
    from robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.runners.teacher_policy_runner import (
        TeacherPolicyRunner,
    )
    TEACHER_POLICY_AVAILABLE = True
except ImportError:
    TEACHER_POLICY_AVAILABLE = False
    print(
        "[WARNING] TeacherPolicyRunner not available. "
        "Using default OnPolicyRunner."
    )
    
def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # make a smaller scene for play
    env_cfg.scene.num_envs = 50
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        controller = Se2Keyboard(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    use_teacher_policy = (
        TEACHER_POLICY_AVAILABLE
        and "Unitree-Go2-v0" in args_cli.task
        and hasattr(agent_cfg, "use_teacher_policy")
        and agent_cfg.use_teacher_policy
    )

    if use_teacher_policy:
        print("[INFO] Using TeacherPolicyRunner for play.")
        runner = TeacherPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        print("[INFO] Using OnPolicyRunner for play.")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    
    # # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner.load(resume_path)

    # # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # # extract the neural network module
    # # we do this in a try-except to maintain backwards compatibility.
    # try:
    #     # version 2.3 onwards
    #     policy_nn = ppo_runner.alg.policy
    # except AttributeError:
    #     # version 2.2 and below
    #     policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    # 统一获取 normalizer（TeacherPolicyRunner / OnPolicyRunner 都有）
    normalizer = getattr(runner, "obs_normalizer", None)

    try:
        export_policy_as_onnx(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        export_policy_as_jit(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        print(f"[INFO] Exported policy to: {export_model_dir}")
    except Exception as e:
        print(f"[WARN] Export failed: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    if use_teacher_policy and hasattr(runner, "history_steps"):
        base0 = obs[:, :-17]
        # 重新初始化并填充
        runner._init_history_buffer()
        for _ in range(runner.history_steps):
            runner._update_history_buffer(base0)
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # policy 已封装: 内部会截取 base_obs -> 更新 history -> 预测 collision -> 拼接 -> 归一化 -> act
            actions = policy(obs)

            # 环境步进
            obs, _, _, _ = env.step(actions)

            # 调试前 5 步
            if timestep < 5:
                print(f"[DBG] step={timestep} obs_mean={obs.mean():.3f} obs_std={obs.std():.3f}")
                if use_teacher_policy:
                    # 当前缓冲里最新一帧已在 policy 调用中更新
                    try:
                        collision_pred_dbg = runner.collision_estimator(runner.obs_history_buffer)
                        print("[DBG] collision_pred[0][:8]:", collision_pred_dbg[0, :8].detach().cpu().numpy())
                        print("[DBG] last17 placeholder raw obs (环境输出):", obs[0, -17:].detach().cpu().numpy())
                    except Exception as e:
                        print(f"[DBG] collision pred debug failed: {e}")
                print(f"[DBG] act_mean={actions.mean():.3f} act_std={actions.std():.3f}")
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            rsl_rl_utils.camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
