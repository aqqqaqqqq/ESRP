# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from multiprocessing import freeze_support

# from .data.dataset import StepDataset
# from .model.MobileNet import CNNOnlyPolicy

# from torch.cuda.amp import autocast, GradScaler

# # 1. 初始化
# scaler = GradScaler()
# torch.backends.cudnn.benchmark = True

# # 2. 训练循环
# for epoch in range(epochs):
#     for x_small, x_large, action, fetch in train_loader:
#         x_small, x_large, action, fetch = [t.cuda(non_blocking=True) for t in (x_small, x_large, action, fetch)]
#         optimizer.zero_grad()
#         with autocast():
#             out = model(x_small, x_large, fetch)
#             loss = criterion(out, action)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()


# def main():
#     # 配置
#     device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_root  = "imitation_data/imitation_data_train"
#     val_root    = "imitation_data/imitation_data_val"
#     num_actions = 6
#     batch_size  = 2048
#     lr_backbone = 1e-5
#     lr_head     = 5e-4
#     epochs      = 300

#     writer = SummaryWriter(log_dir="new_runs/mobilev3_no_lstm")

#     # 数据集
#     train_ds = StepDataset(train_root)
#     val_ds   = StepDataset(val_root)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

#     # 模型 & 优化器
#     model = CNNOnlyPolicy(num_actions=num_actions, feature_dim=256, pretrained=True).to(device)
#     optimizer = optim.Adam([
#         {'params': model.fe_s.parameters(), 'lr': lr_backbone},
#         {'params': model.fe_l.parameters(), 'lr': lr_backbone},
#         {'params': model.fetch_emb.parameters(), 'lr': lr_head},
#         {'params': model.fc.parameters(), 'lr': lr_head},
#     ])

#     best_val_loss = float('inf')

#     for epoch in range(1, epochs+1):
#         # —— 训练 ——
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
#         for img_s, img_l, act, fetch in pbar:
#             img_s = img_s.to(device)
#             img_l = img_l.to(device)
#             act   = act.to(device)
#             fetch = fetch.to(device)

#             # 前向
#             probs = model(img_s, img_l, fetch)  # [B, num_actions]
#             loss = F.cross_entropy(probs, act)

#             # 反向 & 更新
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 统计
#             train_loss += loss.item()
#             preds = probs.argmax(dim=1)
#             train_correct += (preds == act).sum().item()
#             train_total   += act.size(0)
#             pbar.set_postfix(
#                 loss=f"{train_loss/len(train_loader):.4f}",
#                 acc=f"{train_correct/train_total:.4f}"
#             )

#         train_loss_epoch = train_loss / len(train_loader)
#         train_acc_epoch  = train_correct / train_total

#         # —— 验证 ——
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
#             for img_s, img_l, act, fetch in pbar:
#                 img_s = img_s.to(device)
#                 img_l = img_l.to(device)
#                 act   = act.to(device)
#                 fetch = fetch.to(device)

#                 probs = model(img_s, img_l, fetch)
#                 loss = F.cross_entropy(probs, act)

#                 val_loss += loss.item()
#                 preds = probs.argmax(dim=1)
#                 val_correct += (preds == act).sum().item()
#                 val_total   += act.size(0)
#                 pbar.set_postfix(
#                     loss=f"{val_loss/len(val_loader):.4f}",
#                     acc=f"{val_correct/val_total:.4f}"
#                 )

#         val_loss_epoch = val_loss / len(val_loader)
#         val_acc_epoch  = val_correct / val_total

#         print(f"Epoch {epoch}/{epochs} | "
#               f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} | "
#               f"Val   loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f}")

#         if val_loss_epoch < best_val_loss:
#             best_val_loss = val_loss_epoch
#             torch.save(model.state_dict(),
#                        "./omnigibson/baseline/IL/checkpoint/best_cnn_policy_mobile_new_v3_no_lstm.pth")

#         writer.add_scalar('Loss/train', train_loss_epoch, epoch)
#         writer.add_scalar('Acc/train', train_acc_epoch, epoch)
#         writer.add_scalar('Loss/val',   val_loss_epoch,   epoch)
#         writer.add_scalar('Acc/val',    val_acc_epoch,    epoch)

#     writer.close()
#     print("训练完成，最佳验证损失：", best_val_loss)

# if __name__ == '__main__':
#     freeze_support()
#     main()

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
from torch.cuda.amp import GradScaler, autocast

from .data.dataset import StepDataset
from .model.MobileNet import CNNOnlyPolicy

def main():
    # —— 基础配置 —— #
    torch.backends.cudnn.benchmark = True  # 让 cuDNN 自动找最优卷积算法
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root  = "imitation_data/imitation_data_train"
    val_root    = "imitation_data/imitation_data_val"
    num_actions = 6
    batch_size  = 2048
    lr_backbone = 1e-5
    lr_head     = 5e-4
    epochs      = 300

    writer = SummaryWriter(log_dir="new_runs/mobilev3_no_lstm")

    # —— 数据加载 —— #
    train_ds = StepDataset(train_root)
    val_ds   = StepDataset(val_root)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # —— 模型 & 优化 —— #
    model = CNNOnlyPolicy(
        num_actions=num_actions, feature_dim=256, pretrained=True
    ).to(device)

    # optional: PyTorch 2.0 compile for speed
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = optim.Adam([
        {'params': model.fe_s.parameters(),   'lr': lr_backbone},
        {'params': model.fe_l.parameters(),   'lr': lr_backbone},
        {'params': model.fetch_emb.parameters(),'lr': lr_head},
        {'params': model.fc.parameters(),     'lr': lr_head},
    ])

    scaler = GradScaler()  # for AMP
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        # —— 训练 —— #
        model.train()
        train_loss = train_correct = train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for img_s, img_l, act, fetch in pbar:
            img_s = img_s.to(device, non_blocking=True)
            img_l = img_l.to(device, non_blocking=True)
            act   = act.to(device,   non_blocking=True)
            fetch = fetch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():  # mixed precision
                probs = model(img_s, img_l, fetch)
                loss  = F.cross_entropy(probs, act)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss   += loss.item()
            preds         = probs.argmax(dim=1)
            train_correct += (preds == act).sum().item()
            train_total   += act.size(0)
            pbar.set_postfix(
                loss=f"{train_loss/len(train_loader):.4f}",
                acc =f"{train_correct/train_total:.4f}"
            )

        train_loss_epoch = train_loss / len(train_loader)
        train_acc_epoch  = train_correct / train_total

        # —— 验证 —— #
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for img_s, img_l, act, fetch in pbar:
                img_s = img_s.to(device, non_blocking=True)
                img_l = img_l.to(device, non_blocking=True)
                act   = act.to(device,   non_blocking=True)
                fetch = fetch.to(device, non_blocking=True)

                with autocast():
                    probs = model(img_s, img_l, fetch)
                    loss  = F.cross_entropy(probs, act)

                val_loss    += loss.item()
                preds        = probs.argmax(dim=1)
                val_correct += (preds == act).sum().item()
                val_total   += act.size(0)
                pbar.set_postfix(
                    loss=f"{val_loss/len(val_loader):.4f}",
                    acc =f"{val_correct/val_total:.4f}"
                )

        val_loss_epoch = val_loss / len(val_loader)
        val_acc_epoch  = val_correct / val_total

        print(f"Epoch {epoch}/{epochs} | "
              f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} | "
              f"Val   loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f}")

        # 保存最优模型
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(
                model.state_dict(),
                "./omnigibson/baseline/IL/checkpoint/best_cnn_policy_mobile_new_v3_no_lstm.pth"
            )

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss_epoch, epoch)
        writer.add_scalar('Acc/train',  train_acc_epoch,  epoch)
        writer.add_scalar('Loss/val',   val_loss_epoch,   epoch)
        writer.add_scalar('Acc/val',    val_acc_epoch,    epoch)

    writer.close()
    print("训练完成，最佳验证损失：", best_val_loss)

if __name__ == '__main__':
    freeze_support()
    main()
