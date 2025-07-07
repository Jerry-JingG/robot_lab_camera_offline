import torch
import torch.nn as nn
import torch.nn.functional as F
from .collision_domain_encoder import CollisionDomainEncoder

class TeacherPolicy(nn.Module):
    """
    教师策略网络:
    能访问特权信息（真实环境碰撞域）的 Actor-Critic 策略。
    教师策略在训练阶段1通过加强学习训练，学会利用真实障碍物分布进行决策。
    架构:
    - 包含一个CollisionDomainEncoder子模块，将真实碰撞域网格编码为潜在特征。
    - Actor网络根据本体观测和编码的特权特征输出动作；
    - Critic网络估计状态值。
    训练:
    - 阶段1采用PPO等算法训练教师策略，通过访问真实碰撞域，提高策略性能。
    - 同时利用真实碰撞标签训练碰撞检测网络φ（可在此网络外部并行训练，或通过辅助损失一起训练）。
    在train.py中，教师策略用于产生专家行为，并保存用于阶段2学生模仿学习。
    """
    def __init__(self, proprio_obs_dim: int, action_dim: int,
                 grid_size: tuple = (20, 20), latent_dim: int = 64,
                 actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 128]):
        """
        初始化教师策略网络。
        参数:
        - proprio_obs_dim: 本体观测维度。
        - action_dim: 动作空间维度。
        - grid_size: 碰撞域网格尺寸 (与环境定义一致)。
        - latent_dim: 碰撞域编码器输出潜在特征维度。
        - actor_hidden_dims: actor MLP隐藏层尺寸列表。
        - critic_hidden_dims: critic MLP隐藏层尺寸列表。
        """
        super(TeacherPolicy, self).__init__()
        # 初始化碰撞域编码器（将占据网格 -> latent特征）
        self.collision_domain_encoder = CollisionDomainEncoder(grid_size=grid_size, latent_dim=latent_dim)
        # Actor网络：输入 = 本体obs + 碰撞域latent，输出 = 动作参数
        actor_layers = []
        actor_input_dim = proprio_obs_dim + latent_dim
        prev_dim = actor_input_dim
        for dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, dim))
            actor_layers.append(nn.ReLU())
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor_net = nn.Sequential(*actor_layers)
        # Critic网络：结构类似，但输出为1维状态值
        critic_layers = []
        critic_input_dim = proprio_obs_dim + latent_dim
        prev_dim = critic_input_dim
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, dim))
            critic_layers.append(nn.ReLU())
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor, collision_grid: torch.Tensor):
        """
        前向传播:
        输入:
        - obs: 当前时间步本体观测 (B, proprio_obs_dim)。
        - collision_grid: 当前时间步机器人周围真实碰撞域网格 (B, 1, H, W)，由环境提供的特权信息。
        输出:
        - action_mean: 动作均值输出 (B, action_dim)。
        - value: 状态价值 (B, 1)。
        流程:
        1. 使用CollisionDomainEncoder将collision_grid编码为latent特征 (B, latent_dim)。
        2. 将obs与latent特征拼接作为Actor-Critic输入。
        3. 经过actor_net得到动作输出，critic_net得到价值输出。
        训练阶段1:
        - PPO利用action_mean采样动作与环境交互，利用value计算优势函数更新策略。
        - φ碰撞检测网络可在并行模块中训练（利用环境返回的碰撞标签），无需参与forward计算。
        """
        # 编码特权碰撞域信息
        latent = self.collision_domain_encoder(collision_grid)  # (B, latent_dim)
        # 拼接本体观测与latent特征
        policy_input = torch.cat([obs, latent], dim=-1)  # (B, proprio_obs_dim+latent_dim)
        # Actor产生动作输出均值
        action_mean = self.actor_net(policy_input)  # (B, action_dim)
        # Critic产生状态值估计
        value = self.critic_net(policy_input)       # (B, 1)
        return action_mean, value

# 用法:
# 在train.py的教师策略训练阶段:
# collision_grid = env.get_collision_domain_obs()  # 环境返回真实障碍物网格
# action_mean, value = teacher_policy(proprio_obs, collision_grid)
# PPO根据action_mean选择动作，与环境交互获得奖励和下一状态，利用value估计进行策略更新。
# 注: 在该阶段，train.py还会调用collision_estimator来预测碰撞，并利用env给出的实际碰撞标签训练collision_estimator。
