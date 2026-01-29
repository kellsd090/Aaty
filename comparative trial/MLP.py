import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import numpy as np

from database_mit import prepare_ecg_data
from model import seed_everything

class MLPNeuron(nn.Module):
    """
    对称 MLP 单元：LayerNorm + Linear + Activation
    """

    def __init__(self, in_features=187, out_features=187):
        super(MLPNeuron, self).__init__()
        self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()  # 与 KAN 的 Base Function (SiLU) 保持一致

    def forward(self, x):
        # x shape: [Batch, 187]
        return self.activation(self.linear(self.ln(x)))


class ECGMLPModel(nn.Module):
    def __init__(self):
        super(ECGMLPModel, self).__init__()
        # 拓扑结构：1 -> 8 -> 8 -> 5 (每个“神经元”处理 187 维特征)
        # 第一层：1个输入神经元到8个隐藏神经元
        self.layer1 = nn.ModuleList([MLPNeuron(187, 187) for _ in range(8)])

        # 第二层：8个隐藏神经元到8个隐藏神经元
        self.layer2 = nn.ModuleList([MLPNeuron(187, 187) for _ in range(8)])

        # 第三层：8个隐藏神经元到5个输出神经元 (输出降维到 1)
        self.layer3 = nn.ModuleList([MLPNeuron(187, 1) for _ in range(5)])

    def forward(self, x):
        # x: [Batch, 187]

        # Layer 1: 1->8
        # MLP 是全连接，每个输出神经元累加所有输入
        # 由于输入只有1个通道，直接广播
        x1 = [neuron(x) for neuron in self.layer1]  # 8 个 [Batch, 187]

        # Layer 2: 8->8
        x2 = []
        for j in range(8):
            # 模拟全连接：所有输入之和经过第 j 个神经元
            sum_in = torch.stack(x1).sum(dim=0)
            x2.append(self.layer2[j](sum_in))

        # Layer 3: 8->5 (Output)
        x3 = []
        for k in range(5):
            sum_in = torch.stack(x2).sum(dim=0)
            x3.append(self.layer3[k](sum_in))  # 每个输出 [Batch, 1]

        # 拼接成 [Batch, 5]
        out = torch.cat(x3, dim=-1)
        return out


def mlp_balanced_loss(outputs, targets, model, gamma_focal=2.0):
    """
    MLP 对比实验损失函数：只保留 Balanced Focal Loss，剔除 KAN 特有的正则项
    """
    log_p = F.log_softmax(outputs, dim=-1)
    p = torch.exp(log_p)
    pt = p.gather(1, targets.view(-1, 1))
    log_pt = log_p.gather(1, targets.view(-1, 1))
    focal_term = (1 - pt) ** gamma_focal

    if hasattr(model, 'class_weights'):
        batch_weights = model.class_weights.to(outputs.device).gather(0, targets).view(-1, 1)
        ce_loss = (-batch_weights * focal_term * log_pt).mean()
    else:
        ce_loss = (-focal_term * log_pt).mean()

    # MLP 仅使用标准 L2 正则化 (通过 Optimizer 的 weight_decay 实现)
    return ce_loss


def calculate_class_weights(dataloader):
    """
    自动统计类别分布并进行平方根平滑处理
    """
    all_labels = []
    for _, y in dataloader:
        all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)

    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        # 1. 基础频率反比
        raw_w = total / (num_classes * counts[i])
        # 2. 【改进】平方根平滑，防止权重过大破坏整体 Acc
        weights[i] = torch.sqrt(torch.tensor(raw_w))

    # 归一化
    weights = weights / weights.mean()
    return weights


def train_mlp_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    model = ECGMLPModel().to(device)

    # 自动化类别权重统计
    print("统计类别分布中...")
    model.class_weights = calculate_class_weights(train_loader).to(device)
    print(f"MLP 平滑类别权重已设定: {model.class_weights.tolist()}")

    # ================= [新增：初始状态评估] =================
    model.eval()
    init_val_loss = 0
    init_val_correct = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            init_val_loss += mlp_balanced_loss(v_out, vy, model).item()
            init_val_correct += (v_out.argmax(1) == vy).sum().item()

    avg_init_loss = init_val_loss / len(val_loader)
    init_acc = 100 * init_val_correct / len(val_loader.dataset)
    print(f"\n[Initial State] Val Loss: {avg_init_loss:.6f}, Val Acc: {init_acc:.2f}%")
    print("-" * 60)
    # ======================================================

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    epochs = 50
    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        train_bar = tqdm(train_loader, desc=f"MLP Epoch {epoch + 1}/{epochs}")

        for bx, by in train_bar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = mlp_balanced_loss(outputs, by, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == by).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / len(train_loader.dataset):.2f}%")

        # 验证阶段
        model.eval()
        val_correct, val_loss = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += mlp_balanced_loss(v_out, vy, model).item()
                val_correct += (v_out.argmax(1) == vy).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_mlp_baseline.pth")
            print(">>> 最佳模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("Early stopping...")
            break

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%")
    print(f"MLP 总参数量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_mlp_baseline()