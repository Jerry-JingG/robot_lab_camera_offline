import torch
import torch.nn as nn
import torch.nn.functional as F

class CollisionDomainEncoder(nn.Module):
    """
    碰撞域编码器 (三层MLP版本):
    将机器人周围的碰撞扫描器射线数据编码为低维潜在向量。
    使用collision scanner的网格pattern射线数据作为输入，通过三层MLP映射为潜在特征。
    教师策略使用该编码器将特权信息（真实环境碰撞扫描数据）映射为潜在特征，用于决策。
    在训练阶段1，教师策略通过访问真实碰撞扫描数据训练此编码器（与策略网络共同训练）。
    学生策略在训练阶段2不会直接使用真实碰撞扫描，但其"混合想象模型"会尝试估计出类似的潜在特征以模仿教师。
    """
    def __init__(self, num_rays: int, hidden_dims: list = [256, 128, 64]):
        """
        初始化碰撞域编码器。
        参数:
        - num_rays: collision scanner的网格pattern的射线数量（输入维度）
        - hidden_dims: 三层MLP的隐藏层维数列表，默认为[256, 128, 64]
        输入假定为一维射线距离数据 (shape: (batch_size, num_rays))，表示每条射线检测到的距离值。
        """
        super(CollisionDomainEncoder, self).__init__()
        
        self.num_rays = num_rays
        self.hidden_dims = hidden_dims
        
        # 定义三层MLP
        # 第一层：输入维度 -> 第一个隐藏层
        self.fc1 = nn.Linear(num_rays, hidden_dims[0])
        # 第二层：第一个隐藏层 -> 第二个隐藏层  
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # 第三层：第二个隐藏层 -> 输出层
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        
        # 添加dropout层用于正则化
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, collision_rays: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
        输入:
        - collision_rays: 碰撞扫描射线数据张量，形状 (B, num_rays)，表示每条射线的距离值。
        输出:
        - latent_feat: 碰撞域潜在特征张量，形状 (B, hidden_dims[2])，默认为(B, 64)。
        调用场景:
        - 训练阶段1，教师策略每步将真实环境碰撞扫描数据通过此编码器获取latent，用于决策网络输入。
        - 训练阶段2，用于评估学生想象模块输出的特征是否接近教师潜在特征。
        """
        # 第一层MLP + ReLU + Dropout
        x = F.relu(self.fc1(collision_rays))
        x = self.dropout1(x)
        
        # 第二层MLP + ReLU + Dropout  
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # 第三层MLP + ReLU (输出层)
        latent_feat = F.relu(self.fc3(x))
        
        return latent_feat

# 用法示例:
# 在教师策略 forward 中:
# collision_rays = env.get_collision_scanner_data()  # 从环境获取碰撞扫描射线数据 (B, num_rays)
# latent = collision_domain_encoder(collision_rays)  # 获得潜在特征 (B, 64)
# 将 latent 与其他观测拼接后送入教师策略的actor-critic网络。
