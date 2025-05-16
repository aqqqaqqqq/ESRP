# # train_lstm_online.py

# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from multiprocessing import freeze_support

# from .data.dataset_lstm import LSTMDataset

# from .model.MobileNet_lstm import SimpleCNN_LSTM
# from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch):
#     # batch: list of tuples (small:[L,3,128,128], large:[L,3,128,128], act:[L], fetch:[L])
#     small_seq = [item[0] for item in batch]
#     large_seq = [item[1] for item in batch]
#     act_seq   = [item[2] for item in batch]
#     fetch_seq = [item[3] for item in batch]

#     # pad to same T=max_i(L_i)
#     small_padded = pad_sequence(small_seq, batch_first=True)  # [B, Tmax,3,128,128]
#     large_padded = pad_sequence(large_seq, batch_first=True)  # [B, Tmax,3,128,128]
#     # actions: pad with -100 so CrossEntropy(ignore_index=-100) 跳过
#     act_padded   = pad_sequence(act_seq,   batch_first=True, padding_value=-100)  # [B, Tmax]
#     # fetch: pad with 0
#     fetch_padded = pad_sequence(fetch_seq, batch_first=True, padding_value=0)      # [B, Tmax]

#     return small_padded, large_padded, act_padded, fetch_padded

# def main():
#     # 配置
#     device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_root  = "imitation_data_train"
#     val_root    = "imitation_data_val"
#     num_actions = 6
#     batch_size  = 4  
#     lr          = 1e-4
#     epochs      = 300

#     ################################ RESUME
#     # 你已经训练了 100 轮，并把最优权重存在了：
#     # checkpoint_path   = "./omnigibson/baseline/IL/best_lstm_policy_simple.pth"
#     #######################################
#     # TensorBoard writer
#     writer = SummaryWriter(log_dir="new_new_runs/lstm_policy_mobile_new_v3_32")

#     # 数据
#     train_ds = LSTMDataset(train_root)
#     val_ds   = LSTMDataset(val_root)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, collate_fn=collate_fn)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)


#     # 模型 & 优化器
#     # model     = LSTMPolicy(num_actions=num_actions, lstm_hidden=256, pretrained=True).to(device)
#     model     = SimpleCNN_LSTM(num_actions=num_actions, lstm_hidden=256).to(device)
#     ################################ RESUME
#     # state = torch.load(checkpoint_path, map_location=device)
#     # model.load_state_dict(state)
#     # print(f"Loaded model weights from {checkpoint_path}")
#     #######################################
#     # optimizer = optim.Adam(model.parameters(), lr=lr)
#     optimizer = torch.optim.Adam([
#                 {'params': model.fe_s.parameters(), 'lr': 1e-5},
#                 {'params': model.fe_l.parameters(), 'lr': 1e-5},
#                 {'params': model.fetch_emb.parameters(), 'lr': 5e-4},
#                 {'params': model.lstm_cell.parameters(), 'lr': 5e-4},
#                 {'params': model.fc.parameters(), 'lr': 5e-4},
#     ])

#     best_val_loss = float("inf")

#     for epoch in range(1, epochs+1):
#         # —— 训练 —— #
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
#         for seq_s, seq_l, seq_a, seq_f in pbar:
#             # seq_s: [1,T,3,128,128], seq_l: [1,T,3,512,512], seq_a: [1,T]
#             B, T, *_ = seq_s.shape
#             # print("seq_len:", T)
#             seq_s = seq_s.to(device)
#             seq_l = seq_l.to(device)
#             seq_a = seq_a.to(device)
#             seq_f = seq_f.to(device)

#             # 前向
#             hidden = model.init_hidden(batch_size=B, device=device)
#             logits_list = []
#             for t in range(T):
#                 small = seq_s[:, t]
#                 large = seq_l[:, t]
#                 is_fetch_step = seq_f[:, t]
#                 probs, hidden = model.forward_step(small, large, is_fetch_step, hidden)
#                 logits_list.append(probs)
#             probs_seq = torch.stack(logits_list, dim=1)  # [B, T, num_actions]

#             # loss
#             loss = F.cross_entropy(
#                 probs_seq.view(-1, num_actions),
#                 seq_a.view(-1),
#                 ignore_index=-100
#             )

#             # 反向 & 更新
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 统计
#             train_loss += loss.item()
#             preds = probs_seq.argmax(dim=-1)  # [B, T]
#             # train_correct += (preds == seq_a).sum().item()
#             # train_total   += seq_a.numel()
#             mask = seq_a != -100               # [B,Tmax]
#             train_correct += (preds[mask] == seq_a[mask]).sum().item()
#             train_total   += mask.sum().item()

#             # 更新进度条
#             avg_loss = train_loss / len(train_loader)
#             avg_acc  = train_correct / train_total if train_total > 0 else 0.0
#             pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

#         train_loss_epoch = train_loss / len(train_loader)
#         train_acc_epoch  = train_correct / train_total

#         # —— 验证 —— #
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ")
#             for seq_s, seq_l, seq_a, seq_f in pbar:
#                 B, T, *_ = seq_s.shape
#                 seq_s = seq_s.to(device)
#                 seq_l = seq_l.to(device)
#                 seq_a = seq_a.to(device)
#                 seq_f = seq_f.to(device)

#                 hidden = model.init_hidden(batch_size=B, device=device)
#                 logits_list = []
#                 for t in range(T):
#                     small = seq_s[:, t]
#                     large = seq_l[:, t]
#                     is_fetch_step = seq_f[:, t]
#                     probs, hidden = model.forward_step(small, large, is_fetch_step, hidden)
#                     logits_list.append(probs)
#                 probs_seq = torch.stack(logits_list, dim=1)

#                 loss = F.cross_entropy(
#                     probs_seq.view(-1, num_actions),
#                     seq_a.view(-1),
#                     ignore_index=-100
#                 )
#                 val_loss += loss.item()

#                 preds = probs_seq.argmax(dim=-1)
#                 # val_correct += (preds == seq_a).sum().item()
#                 # val_total   += seq_a.numel()
#                 mask = seq_a != -100               # [B,Tmax]
#                 val_correct += (preds[mask] == seq_a[mask]).sum().item()
#                 val_total   += mask.sum().item()

#                 avg_loss = val_loss / len(val_loader)
#                 avg_acc  = val_correct / val_total if val_total > 0 else 0.0
#                 pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

#         val_loss_epoch = val_loss / len(val_loader)
#         val_acc_epoch  = val_correct / val_total

#         # 打印 & 保存最优
#         print(f"Epoch {epoch}/{epochs} | "
#             f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} | "
#             f"Val   loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f}")
#         if val_loss_epoch < best_val_loss:
#             best_val_loss = val_loss_epoch
#             torch.save(model.state_dict(), "./omnigibson/baseline/IL/checkpoint_new/best_lstm_policy_mobile_new_v3_32.pth")

#         # TensorBoard 记录
#         writer.add_scalar('Loss/train', train_loss_epoch, epoch)
#         writer.add_scalar('Loss/val',   val_loss_epoch,   epoch)
#         writer.add_scalar('Acc/train',  train_acc_epoch,  epoch)
#         writer.add_scalar('Acc/val',    val_acc_epoch,    epoch)

#     # 关闭 writer
#     writer.close()
#     print("训练完成，最佳验证损失：", best_val_loss)


# if __name__ == '__main__':
#     freeze_support()
#     main()

# train_lstm_online.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
from torch.cuda.amp import autocast, GradScaler

from .data.dataset_lstm import LSTMDataset
from .model.MobileNet_lstm import SimpleCNN_LSTM
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    small_seq = [item[0] for item in batch]
    large_seq = [item[1] for item in batch]
    act_seq   = [item[2] for item in batch]
    fetch_seq = [item[3] for item in batch]

    small_padded = pad_sequence(small_seq, batch_first=True)  
    large_padded = pad_sequence(large_seq, batch_first=True)  
    act_padded   = pad_sequence(act_seq,   batch_first=True, padding_value=-100)  
    fetch_padded = pad_sequence(fetch_seq, batch_first=True, padding_value=0)      

    return small_padded, large_padded, act_padded, fetch_padded

def main():
    # —— 加速开关 —— #
    torch.backends.cudnn.benchmark = True

    # —— 基本配置 —— #
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root  = "imitation_data_train"
    val_root    = "imitation_data_val"
    num_actions = 6
    batch_size  = 20  
    lr_backbone = 1e-5
    lr_head     = 5e-4
    epochs      = 300

    writer = SummaryWriter(log_dir="new_runs/lstm_mobile_accel")

    # —— 数据加载 —— #
    train_ds = LSTMDataset(train_root)
    val_ds   = LSTMDataset(val_root)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    # —— 模型 & 优化 —— #
    model = SimpleCNN_LSTM(num_actions=num_actions, lstm_hidden=256).to(device)
    # 可选：PyTorch 2.0+ 加速编译
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = optim.Adam([
        {'params': model.fe_s.parameters(),   'lr': lr_backbone},
        {'params': model.fe_l.parameters(),   'lr': lr_backbone},
        {'params': model.fetch_emb.parameters(),'lr': lr_head},
        {'params': model.lstm_cell.parameters(),'lr': lr_head},
        {'params': model.fc.parameters(),     'lr': lr_head},
    ])

    scaler = GradScaler()
    best_val_loss = float("inf")

    for epoch in range(1, epochs+1):
        # —— 训练 —— #
        model.train()
        train_loss = train_correct = train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for seq_s, seq_l, seq_a, seq_f in pbar:
            B, T, *_ = seq_s.shape
            # 非阻塞拷贝
            seq_s = seq_s.to(device, non_blocking=True)
            seq_l = seq_l.to(device, non_blocking=True)
            seq_a = seq_a.to(device, non_blocking=True)
            seq_f = seq_f.to(device, non_blocking=True)

            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size=B, device=device)

            with autocast():
                logits_list = []
                for t in range(T):
                    small = seq_s[:, t]
                    large = seq_l[:, t]
                    is_fetch = seq_f[:, t]
                    probs, hidden = model.forward_step(small, large, is_fetch, hidden)
                    logits_list.append(probs)
                probs_seq = torch.stack(logits_list, dim=1)  # [B, T, num_actions]
                loss = F.cross_entropy(
                    probs_seq.view(-1, num_actions),
                    seq_a.view(-1),
                    ignore_index=-100
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss   += loss.item()
            preds         = probs_seq.argmax(dim=-1)
            mask          = seq_a != -100
            train_correct += (preds[mask] == seq_a[mask]).sum().item()
            train_total   += mask.sum().item()

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
            for seq_s, seq_l, seq_a, seq_f in pbar:
                B, T, *_ = seq_s.shape
                seq_s = seq_s.to(device, non_blocking=True)
                seq_l = seq_l.to(device, non_blocking=True)
                seq_a = seq_a.to(device, non_blocking=True)
                seq_f = seq_f.to(device, non_blocking=True)

                hidden = model.init_hidden(batch_size=B, device=device)
                logits_list = []
                with autocast():
                    for t in range(T):
                        small = seq_s[:, t]
                        large = seq_l[:, t]
                        is_fetch = seq_f[:, t]
                        probs, hidden = model.forward_step(small, large, is_fetch, hidden)
                        logits_list.append(probs)
                    probs_seq = torch.stack(logits_list, dim=1)
                    loss = F.cross_entropy(
                        probs_seq.view(-1, num_actions),
                        seq_a.view(-1),
                        ignore_index=-100
                    )

                val_loss    += loss.item()
                preds        = probs_seq.argmax(dim=-1)
                mask         = seq_a != -100
                val_correct += (preds[mask] == seq_a[mask]).sum().item()
                val_total   += mask.sum().item()

                pbar.set_postfix(
                    loss=f"{val_loss/len(val_loader):.4f}",
                    acc =f"{val_correct/val_total:.4f}"
                )

        val_loss_epoch = val_loss / len(val_loader)
        val_acc_epoch  = val_correct / val_total

        print(f"Epoch {epoch}/{epochs} | "
              f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} | "
              f"Val   loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(
                model.state_dict(),
                "./omnigibson/baseline/IL/new_checkpoint/best_lstm_policy_mobile_accel.pth"
            )

        writer.add_scalar('Loss/train', train_loss_epoch, epoch)
        writer.add_scalar('Acc/train', train_acc_epoch,     epoch)
        writer.add_scalar('Loss/val',   val_loss_epoch,     epoch)
        writer.add_scalar('Acc/val',    val_acc_epoch,      epoch)

    writer.close()
    print("训练完成，最佳验证损失：", best_val_loss)

if __name__ == '__main__':
    freeze_support()
    main()
