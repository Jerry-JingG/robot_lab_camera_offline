# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import rsl_rl

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

from dataclasses import asdict
from rsl_rl.algorithms import PPO
from ..agents.teacher_policy import TeacherPolicy

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
                 **kwargs):
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
        print("=================================================")
        print("TeacherPolicyRunner.__init__ IS CALLED!")
        print("=================================================")

        # 1. 手动调用父类的部分初始化代码，或者直接设置必要的属性
        # 我们不能直接调用 super().__init__，因为它会创建策略
        # 所以我们把 OnPolicyRunner.__init__ 的代码复制过来并修改
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 多GPU配置 (从父类复制)
        self._configure_multi_gpu()

        # 训练类型 (从父类复制)
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # 解析观测维度 (从父类复制)
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        
        # 关键修复：添加这部分缺失的逻辑，用于设置特权观察的类型
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"
            else:
                self.privileged_obs_type = None

        # 关键修复：使用更通用的方式解析特权观测的维度
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # 2. **在这里创建我们自己的 TeacherPolicy 实例**
        # policy_cfg 中包含了 actor_hidden_dims 等所有需要的参数
        policy = TeacherPolicy(
            num_actor_obs=num_obs,
            num_critic_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        print(f"Successfully created policy of type: {type(policy)}")

        # 3. 创建算法实例，并将我们的策略实例传给它
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 4. 完成剩余的初始化 (从父类 OnPolicyRunner.__init__ 复制)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # 初始化经验归一化
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # 初始化存储
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # 日志相关设置
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # 你的自定义初始化
        self.collision_loss_fn = nn.BCELoss()
        self.collision_loss_buffer = deque(maxlen=100)
        
        # TODO: 初始化历史观测缓存
        self.history_buffer = None
        # self._init_history_buffer()
        
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
        # 获取观测维度（policy组的观测维度，即45维）
        obs_dim = 45  # base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(12) + joint_vel(12) + actions(12)
        history_steps = 10  # 历史长度
    
        # 创建历史缓冲区：(num_envs, history_steps, obs_dim)
        self.obs_history_buffer = torch.zeros(
            (self.env.num_envs, history_steps, obs_dim), 
            device=self.device
        )
    
        # 初始化 CollisionEstimator
        from ..agents.moduals.collision_estimator import CollisionEstimator
        self.collision_estimator = CollisionEstimator(
            input_dim=obs_dim,
            history_steps=history_steps,
            num_links=17,  # Go2机器人连杆数
            hidden_dim=64
        ).to(self.device)

    def _update_history_buffer(self, current_obs: torch.Tensor):
        """
        更新历史观测缓存
        
        Args:
            current_obs: 当前观测 (num_envs, obs_dim)
        """
        # 将缓冲区向左移动一位（移除最旧的观测）
        self.obs_history_buffer[:, :-1, :] = self.obs_history_buffer[:, 1:, :]
    
        # 将新观测添加到缓冲区末尾
        self.obs_history_buffer[:, -1, :] = current_obs

    def get_collision_estimation(self) -> torch.Tensor:
        """
        获取当前的碰撞估计
        
        Returns:
            collision_pred: 碰撞概率 (num_envs, 17)
        """
        with torch.no_grad():
            collision_pred = self.collision_estimator(self.obs_history_buffer)
        return collision_pred

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
        # 在训练开始前初始化历史缓冲区
        self._init_history_buffer()
    
        # 调用父类的学习方法，但在其中插入我们的逻辑
        # 这里您需要重写部分逻辑来在每个时间步调用历史更新
    
        for iteration in range(num_learning_iterations):
            # 收集经验数据
            for step in range(self.num_steps_per_env):
                # 获取当前观测（policy组的观测）
                obs, _ = self.env.get_observations()
                policy_obs = obs  # 这是45维的policy观测
                
                # 更新历史缓冲区
                self._update_history_buffer(policy_obs)
                
                # 获取碰撞估计（可用于训练辅助任务或其他用途）
                collision_pred = self.get_collision_estimation()
                
                # 执行动作，获取奖励等（调用父类逻辑）
                # ...
        
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