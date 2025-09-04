# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from typing import Any

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
#test
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# proprio & visual token encoder
from robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.modules.proprio_tokenizer import (
    ProprioTokenizer,
)
from robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.modules.visual_tokenizer import (
    VisualTokenizer,
)
from robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_go2.modules.token_fusion import (
    fuse_tokens,
)

import robot_lab.tasks  # noqa: F401

# Import TeacherPolicyRunner for teacher policy training
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None
    )

    # 
    # 获取 InteractiveScene    
    unwrapped: Any = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is not None:
        # 正确遍历 art 实例
        for name, art in scene.articulations.items():
            # art 是 Articulation 对象，可以访问 body_names
            body_names = art.body_names
            print(f"\n=== articulation '{name}' body_names（共 {len(body_names)} 项） ===", file=sys.stderr)
            for bn in body_names:
                print(" -", bn, file=sys.stderr)
            print("=== end ===\n", file=sys.stderr)

    obs = getattr(unwrapped, "observation_manager", None)
    print("================observations===================")
    print(obs)

    # Demo: extract policy terms [0..5] as proprio features and tokenize them.
    # This does not affect training; it only prints the token shape once.
    try:
        # Compute the current policy observation (concatenated tensor if configured so)
        if obs is None:
            raise RuntimeError("unwrapped env does not expose observation_manager")
        policy_obs = obs.compute_group("policy")  # Tensor[num_envs, D] or dict if not concatenated

        # If not concatenated, flatten in the same order as term listing
        if isinstance(policy_obs, dict):
            # Flatten per-term along the last dim
            ordered_terms = [policy_obs[name].reshape(policy_obs[name].shape[0], -1)
                             for name in obs.active_terms["policy"]]
            policy_obs = torch.cat(ordered_terms, dim=1)

        assert isinstance(policy_obs, torch.Tensor), "Expected concatenated policy obs as Tensor"

        # Build slice offsets from term shapes
        term_shapes = obs.group_obs_term_dim["policy"]  # list[tuple[int, ...]]
        def _numel(shape_tup: tuple[int, ...]) -> int:
            n = 1
            for s in shape_tup:
                n *= int(s)
            return n
        term_lengths = [
            _numel(shape_tup) for shape_tup in term_shapes
        ]

        # first six terms (indices 0..5): base_ang_vel, projected_gravity, velocity_commands,
        # joint_pos, joint_vel, actions
        k = 6
        proprio_dim = sum(term_lengths[:k])
        start_idx = 0
        end_idx = proprio_dim

        # Extract proprio slice for all envs
        proprio_x = policy_obs[:, start_idx:end_idx]

        # Tokenize
        encoder = ProprioTokenizer(in_dim=proprio_dim, hidden_dims=(256, 256), token_dim=128, use_layernorm=True)
        encoder = encoder.to(proprio_x.device)
        with torch.no_grad():
            t_prop = encoder(proprio_x)
        print(f"[DEBUG] Proprio slice dims (first {k} terms): {proprio_dim}; token shape: {tuple(t_prop.shape)}")
    except Exception as e:
        print(f"[WARN] Proprio token demo failed: {e}")

    # Demo: VisualTokenizer minimal check with dummy stacked depth
    try:
        vt = VisualTokenizer()
        dummy_depth = torch.randn(1, 4, 64, 64)  # [B, 4, 64, 64]
        with torch.no_grad():
            vis_tokens = vt(dummy_depth)
        print(f"[DEBUG] VisualTokenizer tokens shape: {tuple(vis_tokens.shape)}")  # expect (1, 16, 128)

        # Optional: fuse with a dummy proprio token to verify the fusion util
        try:
            dummy_prop = torch.randn(1, vis_tokens.shape[-1])  # [B, D]
            fused = fuse_tokens(dummy_prop, vis_tokens)
            print(f"[DEBUG] Fused tokens shape: {tuple(fused.shape)}")  # expect (1, 17, 128)
        except Exception as e:
            print(f"[WARN] Token fusion util demo failed: {e}")
    except Exception as e:
        print(f"[WARN] VisualTokenizer demo failed: {e}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    # Check if we should use TeacherPolicyRunner based on task name
    use_teacher_policy = (
        TEACHER_POLICY_AVAILABLE
        and "Unitree-Go2-v0" in args_cli.task
        and hasattr(agent_cfg, "use_teacher_policy")
        and agent_cfg.use_teacher_policy
    )

    if use_teacher_policy:
        print(
            "[INFO] Using TeacherPolicyRunner for teacher policy "
            "training."
        )
        runner = TeacherPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device
        )
    else:
        print("[INFO] Using default OnPolicyRunner.")
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device
        )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    print(f"[INFO] Starting training with {agent_cfg.algorithm.class_name} algorithm.")
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[misc]
    # close sim app
    simulation_app.close()
