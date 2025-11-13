# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Optional, Dict, Any
import time
import os

import torch
import torch.nn as nn
import rsl_rl

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.algorithms import PPO
from rsl_rl.utils import store_code_state

class TeacherPolicyRunner(OnPolicyRunner):
    def __init__(self,
                 env: VecEnv,
                 train_cfg: Dict[str, Any],
                 log_dir: Optional[str] = None,
                 device: str = 'cpu',
                 history_steps: int = 10,
                 collision_loss_weight: float = 1.0,  # 添加碰撞损失的权重超参数
                 **kwargs):
        """
        Args:
            env: 环境实例
            train_cfg: 训练配置
            log_dir: 日志目录
            device: 计算设备
            history_steps: 历史观测步数
            collision_loss_weight: 碰撞估计损失权重
        """
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        
        print("=================================================")
        print("TeacherPolicyRunner.__init__ IS CALLED!")
        print("=================================================")
        
        self.history_steps = history_steps
        self.collision_loss_weight = collision_loss_weight
    
        # 初始化历史观测缓存
        self._init_history_buffer()
        
        # 初始化碰撞估计模型
        self._init_collision_estimator()
        
    def _init_history_buffer(self):
        """
        初始化历史观测缓存
        """
        # 获取观测维度
        full_obs = self.env.get_observations()[0]
        base_obs_dim = full_obs.shape[1] - 17
        print(f"历史缓存维度: {base_obs_dim}")
        history_steps = self.history_steps
    
        # 创建历史缓冲区：(num_envs, history_steps, base_obs_dim)
        self.obs_history_buffer = torch.zeros(
            (self.env.num_envs, history_steps, base_obs_dim), 
            device=self.device
        )
    
    # 初始化 CollisionEstimator
    def _init_collision_estimator(self):
        """
        初始化碰撞估计器和其优化器
        """
        full_obs = self.env.get_observations()[0]
        base_obs_dim = full_obs.shape[1] - 17
        
        # 使用基础观测维度初始化 CollisionEstimator
        from ..modules.collision_estimator import CollisionEstimator
        self.collision_estimator = CollisionEstimator(
            input_dim=base_obs_dim,
            history_steps=self.history_steps,
            num_links=17,  # Go2机器人连杆数
            hidden_dim=64
        ).to(self.device)
        
        # 为碰撞估计器创建一个独立的优化器
        self.collision_optimizer = torch.optim.Adam(self.collision_estimator.parameters(), lr=1e-3)
        
        print(f"碰撞估计器初始化完成，输入维度: {base_obs_dim}")

    def _update_history_buffer(self, current_obs: torch.Tensor):
        """
        更新历史观测缓存
        
        Args:
            current_obs: 当前观测 (num_envs, obs_dim)
        """
        # 将缓冲区向左移动一位（移除最旧的观测）
        # 使用 .clone() 来避免源和目标内存重叠的 RuntimeError
        self.obs_history_buffer[:, :-1, :] = self.obs_history_buffer[:, 1:, :].clone()
    
        # 将新观测添加到缓冲区末尾
        self.obs_history_buffer[:, -1, :] = current_obs
        
    def get_enhanced_obs(self, obs_data, predictions: torch.Tensor):
        """
        将真实的碰撞预测加入到观测中
        
        Args:
            obs_data: 观测数据，可能是字典或张量
            predictions: 碰撞估计器的17维输出
            
        Returns:
            注入碰撞预测后的观测（保持原始类型）
        """
        base_obs = obs_data[:, :-17]  # 去掉最后17维的占位符
        enhanced_obs = torch.cat([base_obs, predictions], dim=1)  # 拼接真实预测
        # print(f"增强观测维度: {enhanced_obs.shape}")
        return enhanced_obs

    def _extract_base_observations(self, obs):
        """
        从完整观测中提取基础观测（用于历史缓存）
        """
        # 如果是张量，去掉最后17维（collision_predictions占位符）
        base_obs = obs[:, :-17]
        # print(f"基础观测维度: {base_obs.shape}")
        return base_obs
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Args:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否随机初始化episode长度
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    base_obs = self._extract_base_observations(obs)  # 保存原版，用于历史缓存
                    # 1. 更新历史观测缓存(原始观测)
                    self._update_history_buffer(base_obs)
                    # 2. 生成碰撞预测
                    collision_predictions = self.collision_estimator(self.obs_history_buffer)
                    enhanced_obs = self.get_enhanced_obs(obs, collision_predictions)
                    # Sample actions, PPO基于增强观测
                    actions = self.alg.act(enhanced_obs, privileged_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    
                    self.alg.process_env_step(rewards, dones, infos)
                
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    current_collision_predictions = self.collision_estimator(self.obs_history_buffer)
                    if self.privileged_obs_type is not None:
                        enhanced_privileged_obs = self.get_enhanced_obs(
                            privileged_obs, current_collision_predictions
                        )
                    else:
                        enhanced_privileged_obs = self.get_enhanced_obs(
                            obs, current_collision_predictions
                        )
                    self.alg.compute_returns(enhanced_privileged_obs)
                                    
            # --- 碰撞估计器训练模块 ---
            # 1. 准备“标准答案”：生成17维的0/1假数据，模拟真实碰撞标签
            #    (torch.rand(...) > 0.5) 表示碰撞大约50%的概率发生
            dummy_true_collisions = (torch.rand(self.env.num_envs, 17, device=self.device) > 0.5).float()

            # 2. 做出“预测”：将历史观测数据喂给碰撞估计器
            predicted_collisions = self.collision_estimator(self.obs_history_buffer)

            # 3. 计算“差距”：使用二元交叉熵损失函数
            collision_loss = torch.nn.functional.binary_cross_entropy(predicted_collisions, dummy_true_collisions)

            # 4. 更新“第六感”：单独优化碰撞估计器的参数
            self.collision_optimizer.zero_grad()
            collision_loss.backward()
            self.collision_optimizer.step()
            # --- 碰撞估计器训练结束 ---
                                    
            # update policy (PPO策略更新，这是原本的主要训练任务)
            loss_dict = self.alg.update()
            
            # 将我们新的碰撞损失也加入到日志字典中，方便统一记录
            loss_dict["collision_loss"] = collision_loss.item()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    