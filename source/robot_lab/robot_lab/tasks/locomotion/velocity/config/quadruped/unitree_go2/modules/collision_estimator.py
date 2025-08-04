import torch
import torch.nn as nn
import torch.nn.functional as F

class CollisionEstimator(nn.Module):
    """
    碰撞检测网络 φ (Collision Estimator):
    接收过去若干时间步的本体感知观测 (proprioceptive observations)，输出机器人各部位发生碰撞的概率。
    """

    def __init__(self, input_dim: int, history_steps: int, num_links: int, hidden_dim: int = 64):
        """
        初始化碰撞检测网络φ。
        参数:
        - input_dim: 单个时间步的本体观测向量长度。
        - history_steps: 时间历史长度 (例如 10 步)。
        - num_links: 需要预测碰撞的机器人连杆数量（输出维度）。
        - hidden_dim: 隐藏层维度，用于中间特征表示。
        """
        super(CollisionEstimator, self).__init__()
        self.history_steps = history_steps
        self.num_links = num_links
        # 第一层：线性层，对每个时间步的观测进行特征变换
        self.embed = nn.Linear(input_dim, hidden_dim)
        # 卷积层：三个一维卷积层，在时间维度提取序列特征
        # 输入形状在forward中会变为 (batch, hidden_dim, history_steps)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        # 输出层：线性层，将卷积特征映射为各连杆碰撞概率
        self.output_layer = nn.Linear(hidden_dim * history_steps, num_links)
        # 初始化权重（可选）
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.constant_(self.embed.bias, 0)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        输入:
        - obs_history: 张量形状 (B, history_steps, input_dim)，包含过去若干时间步的本体感知观测序列。
        输出:
        - pred: 张量形状 (B, num_links)，每个元素范围在[0,1]，表示对应连杆发生碰撞的概率。
        该函数在每个仿真步被调用：
        - 阶段1训练中，用实际碰撞标签计算二元交叉熵损失，更新φ网络。
        - 阶段2训练中，用训练好的φ提供碰撞概率给学生策略的想象模型。
        """
        # 将输入 reshape 为 (B*history_steps, input_dim)，通过线性层逐步处理每个时间步特征
        B, T, D = obs_history.shape  # B: batch, T:历史长度, D:单步观测维度
        assert T == self.history_steps, "观测历史长度不匹配CollisionEstimator设置"
        x = obs_history.reshape(B * T, D)
        x = F.relu(self.embed(x))
        # 恢复为时间序列形状 (B, T, hidden_dim) 再转为卷积需要的 (B, hidden_dim, T)
        x = x.reshape(B, T, -1).permute(0, 2, 1)  # (B, hidden_dim, T)
        # 卷积层提取时间维度特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 将卷积输出展平成一维向量，再通过输出层得到每个连杆的碰撞概率
        x = x.reshape(B, -1)  # 展平成 (B, hidden_dim * T)
        pred = torch.sigmoid(self.output_layer(x))
        # pred 形状: (B, num_links)，数值在0-1之间
        return pred

