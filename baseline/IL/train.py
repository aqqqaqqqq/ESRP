import os
import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from .model.EfficientNet_b0 import PolicyNet
from .data.dataset import SceneDataset
from .utils.visualization import plot_training_curves
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 


writer = SummaryWriter(log_dir="runs/exp1")

# 配置
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root  = "data/"     
train_root  = 'imitation_data_train'
val_root    = 'imitation_data_val'    # 场景根目录
num_actions = 6
batch_size  = 32
lr          = 1e-4
epochs      = 20
# val_ratio   = 0.2

# —— 数据加载 —— #
train_ds = SceneDataset(train_root)
val_ds   = SceneDataset(val_root)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

# —— 模型 & 优化器 —— #
model     = PolicyNet(num_actions=num_actions, pretrained=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# —— 训练循环 —— #
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_val_loss = float('inf')
best_val_acc = float('inf')

for epoch in range(1, epochs+1):
    # —— 训练 —— #
    model.train()
    train_loss = train_correct = train_total = 0
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
    for small_img, large_img, labels in train_iter:
        small_img, large_img, labels = small_img.to(device), large_img.to(device), labels.to(device)
        probs = model(small_img, large_img)
        loss = F.nll_loss(torch.log(probs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计
        batch_size_curr = small_img.size(0)
        train_loss    += loss.item() * batch_size_curr
        preds          = probs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += batch_size_curr

        # 更新进度条后缀
        curr_avg_loss = train_loss / train_total
        curr_acc      = train_correct / train_total
        train_iter.set_postfix(loss=f"{curr_avg_loss:.4f}", acc=f"{curr_acc:.4f}")
    train_loss = train_loss / train_total
    train_acc  = train_correct / train_total

    # —— 验证 —— #
    model.eval()
    val_loss = val_correct = val_total = 0
    val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False)
    with torch.no_grad():
        for small_img, large_img, labels in val_iter:
            small_img, large_img, labels = small_img.to(device), large_img.to(device), labels.to(device)
            probs = model(small_img, large_img)
            loss  = F.nll_loss(torch.log(probs), labels)

            batch_size_curr = small_img.size(0)
            val_loss    += loss.item() * batch_size_curr
            preds        = probs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += batch_size_curr

            # 更新进度条后缀
            curr_avg_loss = val_loss / val_total
            curr_acc      = val_correct / val_total
            val_iter.set_postfix(loss=f"{curr_avg_loss:.4f}", acc=f"{curr_acc:.4f}")

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total

    # 记录与输出
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch}/{epochs} "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    # 保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "./omnigibson/baseline/IL/best_model_2.pth")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val',   val_loss,   epoch)
    writer.add_scalar('Acc/train',  train_acc,  epoch)
    writer.add_scalar('Acc/val',    val_acc,    epoch)


# 可视化训练曲线
# plot_training_curves(history, save_path="training")

# 结束后关闭 TensorBoard writer
writer.close()

print("训练完成，最佳验证损失：", best_val_loss)
