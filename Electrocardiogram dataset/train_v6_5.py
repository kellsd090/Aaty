import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import numpy as np

from database_mit import prepare_ecg_data
from model import ECGKANModel, seed_everything, SplineWeightLayer, FastKANLayer


def kan_optimal_loss(outputs, targets, model, epoch,
                     lamb=2e-4,  # L1 稀疏化：压低背景权重
                     mu=4e-2,  # 熵正则：越高，路径越细、越稀疏
                     gamma_smth=4e-2,  # 样条平滑：保证激活函数曲线不抖动
                     mu_att=1e-3,  # 注意力约束：防止 W1 弥散
                     lambda_loc=1e-5,  # [新增] 局部性权重：引导 W2 权重集中在对角线附近
                     lambda_inv=1e-5):  # [新增] 不变性权重：引导 W2 实现权重共享（托普利兹化）

    # 1. 基础分类损失：带权重的交叉熵
    ce_loss = F.cross_entropy(outputs, targets, weight=getattr(model, 'class_weights', None))

    # 2. 拓扑与物理正则化
    reg_loss = 0  # 权重 L1
    entropy_loss = 0  # 突触选择竞争
    smooth_loss = 0  # 样条平滑
    attn_loss = 0  # W1 锐利度
    conv_emulation_loss = 0  # [新增] 卷积等效映射损失

    for m in model.modules():
        # 针对 FastKANLayer 的正则化
        if hasattr(m, 'spline_weights'):
            # L1 惩罚
            reg_loss += torch.sum(torch.abs(m.spline_weights)) / m.num_coeffs
            reg_loss += torch.sum(torch.abs(m.base_scales))

            # 样条平滑惩罚：约束系数二阶差分
            w = m.spline_weights
            diff2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
            smooth_loss += torch.mean(diff2 ** 2)

            # 突触竞争熵正则
            edge_strengths = torch.mean(torch.abs(w), dim=-1) + torch.abs(m.base_scales)
            prob = F.softmax(edge_strengths / 0.1, dim=0)
            entropy_loss += -torch.sum(prob * torch.log(prob + 1e-8)) / m.out_neurons

            # 注意力矩阵 W1 的稀疏约束
            if hasattr(m, 'last_W1') and m.last_W1 is not None:
                attn_loss += torch.mean(m.last_W1 ** 2)

        # 针对变换分支 (W2/b) 的正则化
        if hasattr(m, 'W2'):
            # 基础 L1 约束
            reg_loss += torch.mean(torch.abs(m.W2)) + torch.mean(torch.abs(m.b))

            # --- [核心修改：卷积等效引导逻辑] ---
            W = m.W2  # 形状为 [Hidden_Dim, Hidden_Dim]
            rows, cols = W.shape

            if rows == cols:  # 仅针对方阵执行卷积等效引导
                # A. 局部性约束：惩罚远离对角线的元素 (i-j)^2
                device = W.device
                indices = torch.arange(rows).to(device)
                i_idx, j_idx = torch.meshgrid(indices, indices, indexing='ij')
                dist_mask = (i_idx - j_idx).float() ** 2
                conv_emulation_loss += torch.mean(torch.abs(W) * dist_mask) * lambda_loc

                # B. 平移不变性约束：惩罚对角线上的元素差异 (权重共享)
                if rows > 1:
                    # 比较相邻对角线元素 W[i, j] 与 W[i+1, j+1]
                    diag_diff = W[:-1, :-1] - W[1:, 1:]
                    conv_emulation_loss += torch.mean(diag_diff ** 2) * lambda_inv

    # 3. 动态 Warmup 策略
    warmup = min(1.0, epoch / 5.0)

    # 总损失合成
    # 将 conv_emulation_loss 纳入总 Loss
    total_loss = ce_loss + \
                 warmup * (lamb * reg_loss +
                           mu * entropy_loss +
                           gamma_smth * smooth_loss +
                           mu_att * attn_loss +
                           conv_emulation_loss)

    return total_loss, {
        "ce": ce_loss.item(),
        "ent": entropy_loss.item(),
        "conv_reg": conv_emulation_loss.item() if torch.is_tensor(conv_emulation_loss) else 0
    }


def train_ecg_model(model, train_loader, val_loader, epochs=100, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. 参数分组：强化结构演化动力学
    # 第一组：结构核心参数，给予 10x 学习率打破初始值锁定 (5e-3)
    slow_keys = ['alpha', 'omiga']
    # 第二组：路由与门控参数，给予 2x 学习率 (1e-3)
    mid_keys = ['beta', 'tau', 'temperature', 'gamma']

    params_slow = []
    params_mid = []
    params_base = []

    for name, param in model.named_parameters():
        if any(key in name for key in slow_keys):
            params_slow.append(param)
        elif any(key in name for key in mid_keys):
            params_mid.append(param)
        else:
            params_base.append(param)

    # 2. 构造差异化优化器
    optimizer = optim.AdamW([
        {'params': params_base, 'lr': lr},
        {'params': params_mid, 'lr': lr * 2.0},
        {'params': params_slow, 'lr': lr * 5.0}  # 提升至 10 倍速以对标 SOTA 演化速度
    ], weight_decay=1e-5)

    # 自动计算类别权重以应对不平衡数据集
    model.class_weights = calculate_class_weights(train_loader).to(device)

    # --- 关键修改：替换为余弦退火调度器 ---
    # T_max 为总轮数，使得学习率在训练结束前平滑降低到最低点
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.eval()
    init_correct, init_total, init_v_ce = 0, 0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            _, v_info = kan_optimal_loss(v_out, vy, model, epoch=0)
            init_v_ce += v_info['ce']
            init_total += vy.size(0)
            init_correct += (v_out.argmax(1) == vy).sum().item()

    init_acc = 100 * init_correct / init_total
    print(f"\n[Initial State] Loss: {init_v_ce / len(val_loader):.6f}, Val Acc: {init_acc:.2f}%")
    print("-" * 75)

    best_val_acc = 0.0
    early_stop_counter = 0
    early_stop_patience = 15  # 配合余弦退火，适当增加耐心

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        epoch_ce_loss, epoch_ent_loss, epoch_total_loss = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, info = kan_optimal_loss(outputs, batch_y, model, epoch)
            loss.backward()

            # 梯度裁剪：放宽至 2.0 以允许结构参数有更强的爆发力
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_ce_loss += info['ce']
            epoch_ent_loss += info['ent']
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(ce=f"{info['ce']:.3f}", lr=f"{current_lr:.1e}",
                             acc=f"{100 * train_correct / train_total:.2f}%")

        # 验证逻辑
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_correct += (v_out.argmax(1) == vy).sum().item()
                val_total += vy.size(0)

        val_acc = 100 * val_correct / val_total
        avg_total = epoch_total_loss / len(train_loader)
        avg_ce = epoch_ce_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        print(f"\n[Epoch {epoch + 1} Result] Val Acc: {val_acc:.2f}% | Train Acc: {train_acc:.2f}%")
        print(f"  Avg Loss: {avg_total:.6f} (CE: {avg_ce:.4f})")

        # 核心动力学参数监控
        betas, gammas, alphas = [], [], []
        omiga_means, omiga_stds = [], []
        for m in model.modules():
            if isinstance(m, SplineWeightLayer):
                betas.append(torch.abs(m.beta).item())
                gammas.append(torch.abs(m.gamma).item())
                alphas.append(torch.abs(m.alpha).item())
            if isinstance(m, FastKANLayer):
                omiga_abs = torch.abs(m.omiga)
                omiga_means.append(omiga_abs.mean().item())
                omiga_stds.append(omiga_abs.std().item())

        if betas:
            print(f"  Dynamics: Beta={np.mean(betas):.3f} | Gamma={np.mean(gammas):.3f} | Alpha={np.mean(alphas):.3e}")
        if omiga_means:
            print(f"  Residuals: Omiga Mean={np.mean(omiga_means):.4f} (±{np.mean(omiga_stds):.4f})")

        # 拓扑审计
        conn = model.get_active_conn_info()
        for layer, d in conn.items():
            print(f"  {layer}: {d['active']}/{d['total']} ({d['ratio'] * 100:.1f}%) | tau_avg: {d['tau_mean']:.4f}")

        # --- 修改调度器步进逻辑：余弦退火按 epoch 自动更新 ---
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_kan_v6_dynamic_5.pth")
            print(f"  >>> 参数已更新，保存最佳模型 (Acc: {best_val_acc:.2f}%)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  >>> 验证集未提升 ({early_stop_counter}/{early_stop_patience})")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] 连续 {early_stop_patience} 轮未提升，提前结束。")
            break
        print("-" * 75)


def calculate_class_weights(dataloader, boost_dict={1 : 2.0, 3 : 1.5}):
    all_labels = []
    for _, y in dataloader: all_labels.extend(y.cpu().numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)]).sqrt()
    for idx, factor in boost_dict.items():
        if idx < num_classes:
            weights[idx] = weights[idx] * factor
            print(f">>> 类别 {idx} (S-Premature) 权重已强化: 原权重 x {factor}")
    final_weights = weights / weights.mean()
    return final_weights


if __name__ == "__main__":
    seed_everything(42)
    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)
    model = ECGKANModel()
    train_ecg_model(model, train_loader, val_loader)