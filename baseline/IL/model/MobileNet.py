import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileNetFeatureExtractor(nn.Module):
    """
    使用预训练的 MobileNetV3 Small 提取 (3,H,W) -> 定长特征向量
    """
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features  # 特征层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(mobilenet.classifier[0].in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNOnlyPolicy(nn.Module):
    """
    双路 MobileNetV3 Small 提取特征 -> 拼接 -> 全连接分类，无 LSTM
    """
    def __init__(self, num_actions: int, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.fe_s = MobileNetFeatureExtractor(out_dim=feature_dim, pretrained=pretrained)
        self.fe_l = MobileNetFeatureExtractor(out_dim=feature_dim, pretrained=pretrained)
        self.fetch_emb = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(inplace=True)
        )
        # 2*feature_dim + 2 -> hidden -> num_actions
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

    def forward(self, small: torch.Tensor, large: torch.Tensor, is_fetch: torch.Tensor) -> torch.Tensor:
        # small, large: [B,3,128,128], is_fetch: [B] 或 [B,1]
        f_s = F.relu(self.fe_s(small))
        f_l = F.relu(self.fe_l(large))
        if is_fetch.dim() == 1:
            is_fetch = is_fetch.unsqueeze(1).float()
        fetch_feat = self.fetch_emb(is_fetch)
        fused = torch.cat([f_s, f_l, fetch_feat], dim=1)
        logits = self.fc(fused)
        return F.softmax(logits, dim=1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

if __name__ == "__main__":
    model = CNNOnlyPolicy(num_actions=6, feature_dim=256, pretrained=True)
    total, trainable = model.count_parameters()
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
