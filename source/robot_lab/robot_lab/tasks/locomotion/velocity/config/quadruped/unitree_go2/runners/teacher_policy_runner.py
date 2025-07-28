# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

from dataclasses import asdict
from rsl_rl.algorithms import PPO
from ..agents.teacher_policy import TeacherPolicy

# 导入Teacher Policy相关模块
# from ..agents.teacher_policy import (
#     TeacherPolicy, 
#     CollisionEstimationModel, 
#     CollisionDomainEncoder
# )


class TeacherPolicyRunner(OnPolicyRunner):
    """
    Teacher Policy训练器 - 继承自RSL-RL的OnPolicyRunner
    
    专门用于训练具有特权信息访问能力的Teacher Policy，包括：
    1. 真实碰撞域信息的处理
    2. 碰撞估计模型的联合训练
    3. 历史观测信息的管理
    4. 特权信息的日志记录
    
    训练阶段1的目标：
    - 利用特权信息（真实碰撞域）训练强化学习策略
    - 同时训练碰撞估计模型，为后续蒸馏做准备
    """
    
    def __init__(self,
                 env: VecEnv,
                 train_cfg: Dict[str, Any],
                 log_dir: Optional[str] = None,
                 device: str = 'cpu',
                 history_steps: int = 10,
                 collision_loss_weight: float = 1.0):
        """
        初始化Teacher Policy训练器
        
        Args:
            env: 环境实例
            train_cfg: 训练配置
            log_dir: 日志目录
            device: 计算设备
            history_steps: 历史观测步数
            collision_loss_weight: 碰撞估计损失权重
        """
        # 不再直接调用 super().__init__，因为我们要自定义策略的创建
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        print("=================================================")
        print("TeacherPolicyRunner.__init__ IS CALLED!")
        print("=================================================")

        proprio_obs_dim = self.env.num_obs
        action_dim = self.env.num_actions
        
        # 从字典中获取 actor/critic 隐藏层维度
        actor_hidden_dims = self.policy_cfg['actor_hidden_dims']
        critic_hidden_dims = self.policy_cfg['critic_hidden_dims']
        
        teacher_policy = TeacherPolicy(
            proprio_obs_dim=proprio_obs_dim,
            action_dim=action_dim,
            history_steps=history_steps,
            num_links=4,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        ).to(self.device)

        self.alg = PPO(
            actor_critic=teacher_policy,
            device=self.device,
            **self.alg_cfg  # self.alg_cfg 现在本身就是字典，可以直接解包
        )
        
        if hasattr(teacher_policy, 'num_critic_obs'):
             num_critic_obs = teacher_policy.num_critic_obs
        else:
             print("[WARNING] TeacherPolicy does not have 'num_critic_obs' attribute. Using num_obs as fallback.")
             num_critic_obs = proprio_obs_dim

        # 使用字典键访问
        self.alg.init_storage(self.env.num_envs, self.cfg["num_steps_per_env"], [proprio_obs_dim], [num_critic_obs], [action_dim])

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.current_learning_iteration = 0
        self.writer = None
        
        # TODO: 替换默认的actor_critic为TeacherPolicy
        # self._init_teacher_policy()
        
        # TODO: 初始化历史观测缓存
        self.history_buffer = None
        # self._init_history_buffer()
        
        # TODO: 初始化碰撞估计损失函数
        self.collision_loss_fn = nn.BCELoss()
        
        # TODO: 添加额外的日志记录器
        self.collision_loss_buffer = deque(maxlen=100)
        
    def _init_teacher_policy(self):
        """
        初始化Teacher Policy网络，替换默认的ActorCritic
        """
        # TODO: 实现Teacher Policy的初始化
        # 1. 获取环境的观测和动作维度
        # 2. 创建TeacherPolicy实例
        # 3. 替换self.alg.actor_critic
        pass
    
    def _init_history_buffer(self):
        """
        初始化历史观测缓存
        """
        # TODO: 实现历史观测缓存的初始化
        # 创建形状为 (num_envs, history_steps, obs_dim) 的缓存
        pass
    
    def _update_history_buffer(self, current_obs: torch.Tensor):
        """
        更新历史观测缓存
        
        Args:
            current_obs: 当前观测 (num_envs, obs_dim)
        """
        # TODO: 实现历史观测缓存的更新逻辑
        # 1. 将新观测添加到缓存末尾
        # 2. 移除最旧的观测
        # 3. 处理episode重置的情况
        pass
    
    def _get_collision_domain_from_env(self) -> torch.Tensor:
        """
        从环境中获取真实碰撞域信息（特权信息）
        
        Returns:
            collision_domain: 碰撞域网格 (num_envs, 1, H, W)
        """
        # TODO: 实现从环境中提取碰撞域信息的逻辑
        # 这需要根据具体的环境实现来获取特权信息
        pass
    
    def _get_collision_labels_from_env(self) -> torch.Tensor:
        """
        从环境中获取真实碰撞标签（用于训练碰撞估计模型）
        
        Returns:
            collision_labels: 真实碰撞标签 (num_envs, num_links)
        """
        # TODO: 实现从环境中提取真实碰撞标签的逻辑
        # 这些标签用于监督训练碰撞估计模型
        pass
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        重写学习方法，加入Teacher Policy的特定训练逻辑
        
        Args:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否随机初始化episode长度
        """
        # TODO: 实现Teacher Policy的训练循环
        # 1. 初始化日志记录器
        # 2. 获取初始观测和特权信息
        # 3. 主训练循环：
        #    - 数据收集阶段：使用特权信息进行rollout
        #    - 学习阶段：更新策略网络和碰撞估计模型
        #    - 日志记录：记录额外的碰撞估计损失
        
        # 暂时调用父类方法
        super().learn(num_learning_iterations, init_at_random_ep_len)
    
    def _collect_rollout_with_privileged_info(self):
        """
        使用特权信息进行数据收集
        
        这个方法会：
        1. 获取当前观测和特权信息（碰撞域）
        2. 更新历史观测缓存
        3. 使用Teacher Policy选择动作
        4. 执行环境步进
        5. 收集碰撞标签用于监督学习
        """
        # TODO: 实现带特权信息的rollout收集
        pass
    
    def _update_teacher_policy(self) -> tuple:
        """
        更新Teacher Policy，包括策略网络和碰撞估计模型
        
        Returns:
            tuple: (策略损失, 价值损失, 碰撞估计损失)
        """
        # TODO: 实现Teacher Policy的更新逻辑
        # 1. 计算策略梯度损失（PPO损失）
        # 2. 计算碰撞估计监督损失
        # 3. 组合总损失并反向传播
        # 4. 返回各项损失用于日志记录
        pass
    
    def _compute_collision_estimation_loss(
            self, 
            history_obs: torch.Tensor, 
            true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算碰撞估计模型的监督损失，
        使用历史观测预测碰撞概率与真实标签进行对比
        
        Args:
            history_obs: 历史观测 (batch_size, history_steps, obs_dim)
            true_labels: 真实碰撞标签 (batch_size, num_links)
            
        Returns:
            collision_loss: 碰撞估计损失
        """
        # TODO: 实现碰撞估计损失计算
        # 1. 通过Teacher Policy的碰撞估计模型获取预测概率
        # 2. 使用二元交叉熵损失计算监督损失
        # 3. 返回损失值
        pass
    
    def log(self, locs: Dict[str, Any], width: int = 80, pad: int = 35):
        """
        重写日志记录方法，添加Teacher Policy特定的指标
        
        Args:
            locs: 局部变量字典
            width: 日志宽度
            pad: 填充宽度
        """
        # TODO: 扩展日志记录功能
        # 1. 调用父类的日志记录方法
        # 2. 添加碰撞估计损失的记录
        # 3. 添加特权信息使用情况的记录
        # 4. 添加历史观测缓存状态的记录
        
        # 暂时调用父类方法
        super().log(locs, width, pad)
        
        # TODO: 添加额外的日志记录
        # if hasattr(self, 'collision_loss_buffer') and \
        #    len(self.collision_loss_buffer) > 0:
        #     self.writer.add_scalar(
        #         'Loss/collision_estimation', 
        #         statistics.mean(self.collision_loss_buffer), 
        #         locs['it']
        #     )
    
    def save(self, path: str, 
             infos: Optional[Dict[str, Any]] = None):
        """
        重写保存方法，保存Teacher Policy的完整状态
        
        Args:
            path: 保存路径
            infos: 额外信息
        """
        # TODO: 实现Teacher Policy的保存逻辑
        # 1. 保存策略网络参数
        # 2. 保存碰撞估计模型参数
        # 3. 保存优化器状态
        # 4. 保存训练迭代信息
        # 5. 保存历史缓存状态（如果需要）
        
        # 暂时调用父类方法
        super().save(path, infos)
    
    def load(self, path: str, 
             load_optimizer: bool = True) -> Optional[Dict[str, Any]]:
        """
        重写加载方法，加载Teacher Policy的完整状态
        
        Args:
            path: 加载路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            infos: 加载的额外信息
        """
        # TODO: 实现Teacher Policy的加载逻辑
        # 1. 加载策略网络参数
        # 2. 加载碰撞估计模型参数
        # 3. 加载优化器状态（如果需要）
        # 4. 恢复训练迭代信息
        # 5. 恢复历史缓存状态（如果需要）
        
        # 暂时调用父类方法
        return super().load(path, load_optimizer)
    
    def get_inference_policy(self, device: Optional[str] = None):
        """
        获取用于推理的策略网络
        
        Args:
            device: 目标设备
            
        Returns:
            inference_policy: 推理策略函数
        """
        # TODO: 实现Teacher Policy的推理接口
        # 1. 切换到评估模式
        # 2. 移动到指定设备
        # 3. 返回推理函数，该函数能够处理特权信息
        
        # 暂时调用父类方法
        return super().get_inference_policy(device)
    
    def evaluate_collision_estimation(
            self, 
            test_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        评估碰撞估计模型的性能
        
        Args:
            test_data: 测试数据，包含历史观测和真实标签
            
        Returns:
            metrics: 评估指标字典（准确率、精确率、召回率等）
        """
        # TODO: 实现碰撞估计模型的评估逻辑
        # 1. 使用测试数据进行前向传播
        # 2. 计算分类指标（准确率、精确率、召回率、F1分数）
        # 3. 返回评估结果
        pass