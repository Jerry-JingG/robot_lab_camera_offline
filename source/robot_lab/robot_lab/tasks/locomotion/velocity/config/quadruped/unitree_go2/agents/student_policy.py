import torch
import torch.nn as nn
import torch.nn.functional as F
from .collision_estimator import CollisionEstimator

class HybridImaginationModel(nn.Module):
    """
    混合想象模型 (Hybrid Imagination Model):
    学生策略的关键子模块，利用本体观测历史和碰撞估计结果，推理环境中障碍物的潜在特征。
    实现为带门控循环单元 (GRU) 的序列模型，以整合时间信息。
    - 输入: 当前步本体观测和当前步的碰撞估计概率。
    - 内部: GRU累积历史信息以更新对障碍物布局的想象（隐式估计特征）。
    - 输出: 当前时刻的障碍物隐式特征向量 (latent)，用于学生策略决策。
    训练阶段2中，HybridImaginationModel在学生策略中训练，其输出通过监督学习逼近教师策略的碰撞域编码特征。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        """
        初始化混合想象模型。
        参数:
        - input_dim: GRU每步输入大小 = 本体观测特征维度 + 碰撞估计输出维度。
        - hidden_dim: GRU隐藏状态维度。
        - latent_dim: 输出潜在特征维度（通常可取与CollisionDomainEncoder输出相同，以便对齐教师学生特征）。
        """
        super(HybridImaginationModel, self).__init__()
        self.hidden_dim = hidden_dim
        # GRU层：处理序列输入（这里每一步处理一个时间步，但我们通过maintain隐状态跨步传递）
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 输出映射：将GRU隐藏状态映射为所需的潜在特征维度
        self.fc = nn.Linear(hidden_dim, latent_dim)
        # 初始隐藏状态
        self.reset_hidden_state()

    def reset_hidden_state(self, batch_size: int = 1):
        """重置GRU的隐藏状态。在新episode开始或需要手动重置记忆时调用。"""
        # GRU的hidden_state形状：(num_layers, batch, hidden_dim)
        self.h = torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, input_vec: torch.Tensor):
        """
        前向传播:
        输入:
        - input_vec: 张量形状 (B, input_dim)，当前时间步的输入特征（包含当前本体观测特征和碰撞估计）。
        输出:
        - latent: 张量形状 (B, latent_dim)，当前时间步估计的障碍物潜在特征。
        说明:
        在学生策略ActorCritic中，每个时间步调用一次:
         1. 组合本体观测和碰撞估计为input_vec。
         2. 通过GRU更新隐状态 (使用上一步的隐藏状态 self.h)。
         3. 将当前GRU输出隐藏状态经过全连接映射为latent特征。
        注意: 由于仿真同步调用，我们batch_first=True且每次输入序列长度1，这里直接调用GRUCell等效。
        """
        # 将input_vec视作长度1的序列，更新GRU隐藏状态
        out, self.h = self.gru(input_vec.unsqueeze(1), self.h)  # input shape (B, seq_len=1, input_dim)
        # out形状: (B, seq_len=1, hidden_dim)，取out[:, -1]即 (B, hidden_dim)
        out = out[:, -1, :]
        latent = F.relu(self.fc(out))  # 映射为潜在特征维度，并用ReLU激活
        return latent

class StudentPolicy(nn.Module):
    """
    学生策略 Actor-Critic 网络:
    只能访问有限感知（本体传感 + 碰撞估计）的策略网络，使用混合想象模型来推断障碍物环境特征。
    - Actor: 根据当前本体观测和混合想象模型输出的潜在特征，输出动作分布（例如期望关节速度或力矩）。
    - Critic: 估计当前状态值（用于PPO中评价）。
    结构:
    * 包含预训练的 CollisionEstimator φ（通常已在阶段1训练好并冻结）。
    * 内部集成 HybridImaginationModel (GRU) 保持历史，产生环境latent特征。
    * 将当前本体观测和latent特征一起输入Actor和Critic的MLP。
    调用:
    - 在训练阶段2，PPO算法每步调用 StudentPolicy.forward(obs) 获取动作和价值。
    - 在推理 (play.py) 中，使用StudentPolicy来决定机器人动作（不需要教师特权信息）。
    """
    def __init__(self, proprio_obs_dim: int, action_dim: int, num_links: int,
                 imagination_hidden: int = 128, imagination_latent: int = 64,
                 actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 128]):
        """
        初始化学生策略网络。
        参数:
        - proprio_obs_dim: 本体状态观测维度（不含特权信息）。
        - action_dim: 动作空间维度（策略输出维度）。
        - num_links: 碰撞估计输出维度（机器人连杆数量）。
        - imagination_hidden: 想象模型GRU隐藏层维度。
        - imagination_latent: 想象模型输出潜在特征维度。
        - actor_hidden_dims: actor部分MLP的隐藏层尺寸列表。
        - critic_hidden_dims: critic部分MLP的隐藏层尺寸列表。
        """
        super(StudentPolicy, self).__init__()
        # 初始化碰撞检测网络φ (假定预训练权重将被加载)，并冻结参数以防训练阶段2更新
        self.collision_estimator = CollisionEstimator(input_dim=proprio_obs_dim, history_steps=10, num_links=num_links)
        for param in self.collision_estimator.parameters():
            param.requires_grad = False  # 学生阶段不训练φ，只推理
        
        # 混合想象模型：输入大小 = proprio_obs_dim + num_links（本体+碰撞估计）
        imagination_input_dim = proprio_obs_dim + num_links
        self.imagination_model = HybridImaginationModel(input_dim=imagination_input_dim,
                                                       hidden_dim=imagination_hidden,
                                                       latent_dim=imagination_latent)
        # Actor网络：将 (本体观测 + 想象latent) 作为输入，输出动作分布参数
        actor_layers = []
        actor_input_dim = proprio_obs_dim + imagination_latent
        prev_dim = actor_input_dim
        for dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, dim))
            actor_layers.append(nn.ReLU())
            prev_dim = dim
        # 输出动作层，假设连续动作用参数化高斯策略，则输出为均值；这里先输出均值
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor_net = nn.Sequential(*actor_layers)
        # Critic网络：结构类似actor，但输出为状态价值估计
        critic_layers = []
        critic_input_dim = proprio_obs_dim + imagination_latent
        prev_dim = critic_input_dim
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, dim))
            critic_layers.append(nn.ReLU())
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor, obs_history: torch.Tensor):
        """
        前向传播:
        输入:
        - obs: 当前时间步的本体观测 (tensor shape: (B, proprio_obs_dim))
        - obs_history: 本体观测历史序列 (tensor shape: (B, history_steps, proprio_obs_dim))，用于碰撞估计。
        输出:
        - action_mean: 动作均值输出 (B, action_dim) （这里简化为actor直接输出均值，可按需要扩展为分布参数）
        - value: 状态价值估计 (B, 1)
        逻辑:
        1. 使用历史观测通过collision_estimator预测当前碰撞概率 (φ网络)。
        2. 将当前obs与预测碰撞概率拼接，输入imagination_model，更新GRU得到障碍物latent特征。
        3. 将当前obs与latent特征拼接，一并输入actor_net和critic_net，得到动作和价值。
        在训练阶段2:
        - PPO算法会获取action_mean用于采样动作，value用于计算优势。
        - 同时可以将imagination_model输出的latent与教师的collision_domain_encoder输出进行对比，计算监督损失指导imagination_model训练（模仿教师特征）。
        """
        # 1. 碰撞估计网络预测当前碰撞概率分布
        collision_pred = self.collision_estimator(obs_history)  # (B, num_links)
        # 2. 混合想象模型：将当前obs和collision_pred拼接作为输入，得到环境latent特征
        inp = torch.cat([obs, collision_pred], dim=-1)  # (B, proprio_obs_dim + num_links)
        latent = self.imagination_model(inp)  # (B, imagination_latent)
        # 3. 策略actor-critic计算
        # 拼接当前本体obs与latent作为策略输入
        policy_input = torch.cat([obs, latent], dim=-1)  # (B, proprio_obs_dim + latent_dim)
        # Actor网络输出动作（这里假设动作是连续的均值，如果是离散动作则需要softmax等）
        action_mean = self.actor_net(policy_input)  # (B, action_dim)
        # Critic网络输出状态值估计
        value = self.critic_net(policy_input)       # (B, 1)
        return action_mean, value

# 用法:
# 在train.py的学生策略训练循环中，每步环境返回obs和历史obs:
# action_mean, value = student_policy(obs, obs_hist)
# （然后依据action_mean采样执行动作，计算PPO损失；并可利用teacher_policy产生的专家动作或特征计算额外监督损失）。
# 在play.py中，仅使用student_policy根据实时obs决策动作:
# action_mean, _ = student_policy(obs, obs_hist); 执行action_mean作为机器人动作。
