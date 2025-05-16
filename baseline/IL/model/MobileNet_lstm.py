import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# class MobileNetFeatureExtractor(nn.Module):
#     """
#     使用预训练的 MobileNetV2 提取 (3,H,W) -> 定长特征向量
#     """
#     def __init__(self, out_dim: int = 256, pretrained: bool = True):
#         super().__init__()
#         # 加载预训练 MobileNetV2
#         mobilenet = models. mobilenet_v3_small(pretrained=pretrained)
#         # 保留特征提取部分
#         self.features = mobilenet.features  # [B, 1280, H/32, W/32]
#         # 全局平均池化
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # 线性映射到目标维度
#         self.fc = nn.Linear(1280, out_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B,3,H,W]
#         x = self.features(x)
#         x = self.avgpool(x)              # [B,1280,1,1]
#         x = x.view(x.size(0), -1)        # [B,1280]
#         return self.fc(x)                # [B,out_dim]
class MobileNetFeatureExtractor(nn.Module):
    """
    使用预训练的 MobileNetV3-Small 提取 (3,H,W) -> 定长特征向量
    """
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        # 注意这里改成真实的 in_dim
        in_dim = mobilenet.classifier[0].in_features  # 576
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimpleCNN_LSTM(nn.Module):
    """
    双路 MobileNetV2 + is_fetch 特征 -> 拼接 -> LSTMCell -> 分类
    """
    def __init__(
        self,
        num_actions: int,
        feature_dim: int = 256,
        lstm_hidden: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        # 两路 MobileNetV2 特征提取
        self.fe_s = MobileNetFeatureExtractor(out_dim=feature_dim, pretrained=pretrained)
        self.fe_l = MobileNetFeatureExtractor(out_dim=feature_dim, pretrained=pretrained)
        # is_fetch 映射成 2 维
        self.fetch_emb = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(inplace=True)
        )
        # LSTMCell 输入 dim = 2*feature_dim + 2
        self.lstm_cell = nn.LSTMCell(feature_dim * 2 + 2, lstm_hidden)
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
        # small, large: [B,3,128,128]
        # is_fetch: [B] 或 [B,1]
        # hidden: (h_prev, c_prev)
        f_s = F.relu(self.fe_s(small))  # [B,feature_dim]
        f_l = F.relu(self.fe_l(large))  # [B,feature_dim]
        if is_fetch.dim() == 1:
            is_fetch = is_fetch.unsqueeze(1).float()
        fetch_feat = self.fetch_emb(is_fetch)  # [B,2]
        fused = torch.cat([f_s, f_l, fetch_feat], dim=1)  # [B,2*feature_dim+2]
        h_prev, c_prev = hidden
        h_next, c_next = self.lstm_cell(fused, (h_prev, c_prev))
        logits = self.fc(h_next)
        probs = F.softmax(logits, dim=1)
        return probs, (h_next, c_next)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    model = SimpleCNN_LSTM(num_actions=6, lstm_hidden=256, pretrained=True)
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
