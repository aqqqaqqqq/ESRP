import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
from torch.cuda.amp import autocast, GradScaler
import argparse
import os

from omnigibson.baseline.IL.data.dataset_lstm import LSTMDataset
from omnigibson.baseline.IL.model.MobileNet_lstm import SimpleCNN_LSTM
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
    # —— 解析命令行参数 —— #
    parser = argparse.ArgumentParser(description='Train LSTM model for imitation learning')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (used for checkpoint and log directories)')
    parser.add_argument('--train_root', type=str, default='imitation_data_train',
                        help='Root directory for training data')
    parser.add_argument('--val_root', type=str, default='imitation_data_val',
                        help='Root directory for validation data')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (MobileNet)')
    parser.add_argument('--lr_head', type=float, default=5e-4,
                        help='Learning rate for head (LSTM, FC)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--scheduler_patience', type=int, default=8,
                        help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which the learning rate will be reduced')
    parser.add_argument('--scheduler_min_lr_backbone', type=float, default=1e-8,
                        help='Minimum learning rate for backbone')
    parser.add_argument('--scheduler_min_lr_head', type=float, default=1e-7,
                        help='Minimum learning rate for head')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--max_seq_len', type=int, default=64,
                        help='Maximum sequence length for each chunk')
    parser.add_argument('--train_stride', type=int, default=32,
                        help='Stride for training set sliding window (creates overlapping chunks). If None, uses non-overlapping windows')
    
    args = parser.parse_args()
    
    # —— 创建实验目录结构 —— #
    exp_dir = '/home/user/Desktop/il/OmniGibson-Rearrange/omnigibson/baseline/IL/experiments'
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints', args.exp_name)
    log_dir = os.path.join(exp_dir, 'logs', args.exp_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"实验名称: {args.exp_name}")
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"日志目录: {log_dir}")
    
    # —— 加速开关 —— #
    torch.backends.cudnn.benchmark = True

    # —— 基本配置 —— #
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root  = args.train_root
    val_root    = args.val_root
    num_actions = 6
    batch_size  = args.batch_size
    lr_backbone = args.lr_backbone
    lr_head     = args.lr_head
    epochs      = args.epochs

    writer = SummaryWriter(log_dir=log_dir)

    # —— 数据加载 —— #
    # 训练集：使用重叠窗口（stride < max_seq_len）
    # 验证集：使用非重叠窗口（stride = None，即 max_seq_len）
    train_ds = LSTMDataset(train_root, max_seq_len=args.max_seq_len, stride=args.train_stride)
    val_ds   = LSTMDataset(val_root, max_seq_len=args.max_seq_len, stride=None)
    
    print(f"训练集: max_seq_len={args.max_seq_len}, stride={args.train_stride} (重叠窗口)")
    print(f"验证集: max_seq_len={args.max_seq_len}, stride={args.max_seq_len} (非重叠窗口)")
    print(f"训练集chunks数: {len(train_ds)}, 验证集chunks数: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
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

    # —— 学习率调度器（根据验证loss调整）—— #
    # 由于optimizer有多个参数组，我们需要分别为backbone和head创建调度器
    # 创建临时优化器用于调度器（只用于计算新学习率）
    temp_optimizer_backbone = optim.Adam([{'params': model.fe_s.parameters(), 'lr': lr_backbone}])
    temp_optimizer_head = optim.Adam([{'params': model.fc.parameters(), 'lr': lr_head}])
    
    scheduler_backbone = optim.lr_scheduler.ReduceLROnPlateau(
        temp_optimizer_backbone,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr_backbone,
        verbose=False
    )
    
    scheduler_head = optim.lr_scheduler.ReduceLROnPlateau(
        temp_optimizer_head,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr_head,
        verbose=False
    )
    
    def update_lr_from_scheduler(val_loss):
        """根据验证loss更新学习率"""
        lr_changes = {'backbone': None, 'head': None}
        
        # 同步临时优化器的学习率到实际值
        temp_optimizer_backbone.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
        temp_optimizer_head.param_groups[0]['lr'] = optimizer.param_groups[2]['lr']
        
        # 更新backbone学习率
        old_lr_backbone = optimizer.param_groups[0]['lr']
        scheduler_backbone.step(val_loss)
        new_lr_backbone = temp_optimizer_backbone.param_groups[0]['lr']
        if old_lr_backbone != new_lr_backbone:
            # 同步到实际优化器的backbone参数组
            optimizer.param_groups[0]['lr'] = new_lr_backbone
            optimizer.param_groups[1]['lr'] = new_lr_backbone
            lr_changes['backbone'] = (old_lr_backbone, new_lr_backbone)
        
        # 更新head学习率
        old_lr_head = optimizer.param_groups[2]['lr']
        scheduler_head.step(val_loss)
        new_lr_head = temp_optimizer_head.param_groups[0]['lr']
        if old_lr_head != new_lr_head:
            # 同步到实际优化器的head参数组
            for param_group in optimizer.param_groups[2:]:
                param_group['lr'] = new_lr_head
            lr_changes['head'] = (old_lr_head, new_lr_head)
        
        return lr_changes

    scaler = GradScaler()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    start_epoch = 1
    
    # —— 从checkpoint恢复训练 —— #
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 模型权重已加载")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 优化器状态已加载")
            
            # 加载scaler状态
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ GradScaler状态已加载")
            
            # 加载训练状态
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"✓ 从epoch {start_epoch}继续训练")
            
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
                print(f"✓ 最佳验证准确率: {best_val_acc:.4f}")
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print(f"✓ 最佳验证损失: {best_val_loss:.4f}")
            
            # 加载学习率调度器状态（需要手动恢复）
            if 'scheduler_backbone_state' in checkpoint:
                scheduler_backbone.load_state_dict(checkpoint['scheduler_backbone_state'])
                print("✓ Backbone调度器状态已加载")
            
            if 'scheduler_head_state' in checkpoint:
                scheduler_head.load_state_dict(checkpoint['scheduler_head_state'])
                print("✓ Head调度器状态已加载")
            
            # 同步临时优化器的学习率
            if 'lr_backbone' in checkpoint:
                temp_optimizer_backbone.param_groups[0]['lr'] = checkpoint['lr_backbone']
                optimizer.param_groups[0]['lr'] = checkpoint['lr_backbone']
                optimizer.param_groups[1]['lr'] = checkpoint['lr_backbone']
            
            if 'lr_head' in checkpoint:
                temp_optimizer_head.param_groups[0]['lr'] = checkpoint['lr_head']
                for param_group in optimizer.param_groups[2:]:
                    param_group['lr'] = checkpoint['lr_head']
            
            print(f"当前学习率 - Backbone: {optimizer.param_groups[0]['lr']:.2e}, "
                  f"Head: {optimizer.param_groups[2]['lr']:.2e}")
            print("-" * 80)
        else:
            print(f"⚠️  警告: checkpoint文件不存在: {args.resume}")
            print("将从头开始训练")

    for epoch in range(start_epoch, epochs+1):
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

        # —— 更新学习率（根据验证loss）—— #
        lr_changes = update_lr_from_scheduler(val_loss_epoch)
        
        # 记录当前学习率
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[2]['lr']
        
        # 打印学习率变化信息
        lr_info = []
        if lr_changes['backbone']:
            old, new = lr_changes['backbone']
            lr_info.append(f"Backbone LR: {old:.2e}→{new:.2e}")
        if lr_changes['head']:
            old, new = lr_changes['head']
            lr_info.append(f"Head LR: {old:.2e}→{new:.2e}")
        
        lr_str = " | " + ", ".join(lr_info) if lr_info else ""
        print(f"Epoch {epoch}/{epochs} | "
              f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} | "
              f"Val   loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f} | "
              f"LR: backbone={current_lr_backbone:.2e}, head={current_lr_head:.2e}{lr_str}")

        # 更新最佳验证损失
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
        
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            # 保存完整的checkpoint（包含训练状态）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_backbone_state': scheduler_backbone.state_dict(),
                'scheduler_head_state': scheduler_head.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'lr_backbone': optimizer.param_groups[0]['lr'],
                'lr_head': optimizer.param_groups[2]['lr'],
            }, checkpoint_path)
            print(f"✓ 保存最佳模型到: {checkpoint_path}")
        
        # 每个epoch保存一次checkpoint（用于恢复训练）
        # 只保存最近的几个checkpoint，避免占用太多空间
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_backbone_state': scheduler_backbone.state_dict(),
            'scheduler_head_state': scheduler_head.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[2]['lr'],
        }, epoch_checkpoint_path)
        
        # 删除旧的checkpoint（只保留最近的3个）
        if epoch > 3:
            old_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch-3}.pth')
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        writer.add_scalar('Loss/train', train_loss_epoch, epoch)
        writer.add_scalar('Acc/train', train_acc_epoch,     epoch)
        writer.add_scalar('Loss/val',   val_loss_epoch,     epoch)
        writer.add_scalar('Acc/val',    val_acc_epoch,      epoch)
        writer.add_scalar('LR/backbone', current_lr_backbone, epoch)
        writer.add_scalar('LR/head', current_lr_head, epoch)

    writer.close()
    print("=" * 80)
    print(f"训练完成！实验名称: {args.exp_name}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {checkpoint_dir}")
    print(f"日志已保存到: {log_dir}")
    print("=" * 80)

if __name__ == '__main__':
    freeze_support()
    main()
