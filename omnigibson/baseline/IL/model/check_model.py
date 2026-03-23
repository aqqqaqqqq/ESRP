import torch

# 1. 加载 checkpoint
ckpt = torch.load("C:/Users/Admin/Desktop/OmniGibson-Rearrange/omnigibson/baseline/IL/new_checkpoint/best_lstm_policy_mobile_accel.pth", map_location="cpu")

# 2. 打印顶层 keys
print("Keys in checkpoint:", ckpt.keys())

# 3. 取出 state_dict
sd = ckpt.get('model_state_dict', ckpt)

# 4. 打印每层参数名称和形状
for name, param in sd.items():
    print(f"{name:40} → {tuple(param.shape)}")

# 5. 如果保存了整个模型，直接打印结构
if 'model_state_dict' not in ckpt:
    print("/nModel structure:")
    print(ckpt)
