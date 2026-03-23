import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    一个简单的三层卷积网络，把 (3,H,W) -> 一个定长特征向量
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        # conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        # conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3   = nn.BatchNorm2d(128)

        # 全局平均池化到 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 线性层把 128 -> out_dim
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        returns: [B, out_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 下采样 2×
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)     # [B,128,1,1]
        x = x.view(x.size(0), -1)  # [B,128]
        return self.fc(x)         # [B,out_dim]


class SimpleCNN_LSTM(nn.Module):
    """
    双路独立 CNN 提取特征 -> 拼接 -> LSTMCell -> 分类
    """
    def __init__(
        self,
        num_actions: int,
        feature_dim: int = 256,
        lstm_hidden: int = 256
    ):
        super().__init__()
        # 两路简单 CNN
        self.fe_s = CNNFeatureExtractor(out_dim=feature_dim)
        self.fe_l = CNNFeatureExtractor(out_dim=feature_dim)
        # 将 is_fetch (标量) 映射成 4 维
        self.fetch_emb = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(inplace=True)
        )
        # LSTMCell
        self.lstm_cell = nn.LSTMCell(feature_dim * 2 + 4, lstm_hidden)
        # 分类头
        self.fc = nn.Linear(lstm_hidden, num_actions)

    def init_hidden(self, batch_size: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=device)
        return (h0, c0)

    def forward_step(
        self,
        small: torch.Tensor,
        large: torch.Tensor,
        is_fetch: torch.Tensor,
        hidden: tuple
    ):
        """
        small: [B,3,128,128]
        large: [B,3,128,128]  # 已在 Dataset 中 resize
        hidden: (h_prev, c_prev), each [B, lstm_hidden]
        returns: probs [B,num_actions], new_hidden
        """
        # 1) 各自提取特征
        f_s = F.relu(self.fe_s(small))   # [B,feature_dim]
        f_l = F.relu(self.fe_l(large))   # [B,feature_dim]
        # 2) is_fetch 编码
        if is_fetch.dim() == 1:
            is_fetch = is_fetch.unsqueeze(1).float()  # [B,1]
        fetch_feat = self.fetch_emb(is_fetch)        # [B,4]
        # 2) 特征拼接
        fused = torch.cat([f_s, f_l, fetch_feat], dim=1)  # [B, feature_dim*2+4]
        # 3) LSTMCell
        h_prev, c_prev = hidden
        h_next, c_next = self.lstm_cell(fused, (h_prev, c_prev))
        # 4) 分类
        logits = self.fc(h_next)               # [B,num_actions]
        probs  = F.softmax(logits, dim=1)
        return probs, (h_next, c_next)

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":

    model = SimpleCNN_LSTM(num_actions=6, lstm_hidden=256)
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    # visualize_model()