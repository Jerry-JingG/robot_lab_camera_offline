import torch
import torch.nn as nn
import torch.nn.functional as F

class CollisionDomainEncoder(nn.Module):
    """
    显式碰撞域编码器:
    将机器人周围的显式碰撞域表示 (Collision Domain) 编码为低维潜在向量。
    碰撞域定义为以机器人为中心的长方体空间，离散为网格（例如二维栅格表示周围障碍物分布）。
    教师策略使用该编码器将特权信息（真实环境障碍物分布）映射为潜在特征，用于决策。
    在训练阶段1，教师策略通过访问真实碰撞域数据训练此编码器（与策略网络共同训练）。
    学生策略在训练阶段2不会直接使用真实碰撞域，但其“混合想象模型”会尝试估计出类似的潜在特征以模仿教师。
    """
    def __init__(self, grid_size: tuple = (20, 20), hidden_dim: int = 32, latent_dim: int = 64):
        """
        初始化碰撞域编码器。
        参数:
        - grid_size: 碰撞域网格大小 (高度方向可能简化忽略，只考虑水平面网格，例如20x20)。
        - hidden_dim: 卷积通道和中间特征维度。
        - latent_dim: 输出潜在向量维度。
        碰撞域输入假定为二维bool网格 (shape: grid_size)，表示每个网格单元是否存在障碍物。
        """
        super(CollisionDomainEncoder, self).__init__()
        # 假设输入为单通道二维网格
        in_channels = 1
        self.grid_size = grid_size
        # 定义2D卷积层来提取空间特征
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1)
        # 计算卷积后的特征尺寸以确定全连接输入维度
        # 例: 若grid_size=20x20, 经stride2两次 -> 尺寸约5x5 (取整)。
        # 计算：((20 + 2*pad - kernel)//stride + 1)
        # 近似计算输出大小:
        out_h = grid_size[0] // 2 // 2  # 两次stride=2
        out_w = grid_size[1] // 2 // 2
        conv_out_dim = (hidden_dim * 2) * out_h * out_w
        # 输出层：线性层将卷积特征映射为潜在特征向量
        self.fc = nn.Linear(conv_out_dim, latent_dim)
        # 初始化权重
        nn.init.xavier_uniform_(self.conv1.weight); nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight); nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_uniform_(self.conv3.weight); nn.init.constant_(self.conv3.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.constant_(self.fc.bias, 0)

    def forward(self, collision_grid: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
        输入:
        - collision_grid: 碰撞域栅格张量，形状 (B, 1, H, W)，0/1表征网格是否被障碍占据。
        输出:
        - latent_feat: 碰撞域潜在特征张量，形状 (B, latent_dim)。
        调用场景:
        - 训练阶段1，教师策略每步将真实环境障碍物网格通过此编码器获取latent，用于决策网络输入。
        - 训练阶段2，用于评估学生想象模块输出的特征是否接近教师潜在特征（例如可用于监督学习目标）。
        """
        # 卷积提取空间特征
        x = F.relu(self.conv1(collision_grid))   # (B, hidden_dim, H/2, W/2)
        x = F.relu(self.conv2(x))               # (B, 2*hidden_dim, H/4, W/4)
        x = F.relu(self.conv3(x))               # (B, 2*hidden_dim, H/4, W/4) 大小保持
        # 展平并通过全连接获取潜在向量
        x = x.view(x.size(0), -1)
        latent_feat = F.relu(self.fc(x))
        return latent_feat

# 用法示例:
# 在教师策略 forward 中:
# collision_grid = env.get_collision_domain()  # 从环境获取真实障碍物栅格 (B,1,H,W)
# latent = collision_domain_encoder(collision_grid)
# 将 latent 与其他观测一起送入教师策略的决策网络。
