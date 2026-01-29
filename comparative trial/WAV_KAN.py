import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from database_mit import prepare_ecg_data
from model import seed_everything


# --- 优化后的高效小波层：张量并行化计算 ---
class FastWAVLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FastWAVLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 矩阵化参数：每个连接边都有独立的小波尺度、平移和权重
        self.scale = nn.Parameter(torch.ones(in_dim, out_dim))
        self.translation = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.wavelet_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)

    def mexican_hat(self, x):
        """
        批量计算 Mexican Hat 小波响应: psi(t) = (1-t^2) * exp(-t^2/2)
        x shape: [Batch, in_dim]
        """
        x_expanded = x.unsqueeze(-1)  # [Batch, in_dim, 1]
        # 利用广播机制计算所有边的 t 值
        t = (x_expanded - self.translation) / (self.scale + 1e-8)

        # 小波解析式
        psi = (1 - t ** 2) * torch.exp(-(t ** 2) / 2.0)
        return psi

    def forward(self, x):
        # 1. 基础路径 (SiLU 非线性基础)
        base_act = F.silu(x).unsqueeze(-1)
        base_out = (base_act * self.base_weight).sum(dim=1)

        # 2. 小波残差路径
        psi = self.mexican_hat(x)  # [Batch, in_dim, out_dim]
        wavelet_out = (psi * self.wavelet_weight).sum(dim=1)

        return base_out + wavelet_out


class ECGWAVKAN(nn.Module):
    def __init__(self):
        super(ECGWAVKAN, self).__init__()
        # 结构相似性对标：187 -> 32 -> 32 -> 5
        # 虽然不再限制参数对等，但保持层级逻辑的一致性
        self.layer1 = FastWAVLayer(187, 32)
        self.layer2 = FastWAVLayer(32, 32)
        self.layer3 = FastWAVLayer(32, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out


# --- 训练逻辑更新 ---
def train_wav_kan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    model = ECGWAVKAN().to(device)

    # 类别权重统计（包含平方根平滑）
    all_labels = []
    for _, y in train_loader: all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    raw_weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)])
    model.class_weights = torch.sqrt(raw_weights / raw_weights.mean()).to(device)

    # ================= [初始状态评估补全] =================
    model.eval()
    init_val_loss = 0
    init_correct = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            init_val_loss += F.cross_entropy(v_out, vy, weight=model.class_weights).item()
            init_correct += (v_out.argmax(1) == vy).sum().item()

    avg_init_loss = init_val_loss / len(val_loader)
    print(f"\n[Initial State] Loss: {avg_init_loss:.6f}, Val Acc: {100 * init_correct / len(val_loader.dataset):.2f}%")
    print("-" * 75)

    # 2. 训练配置 (使用 AdamW 提升稳定性)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_acc = 0

    for epoch in range(50):
        model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"WAV-KAN Epoch {epoch + 1}/50")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)

            # 使用 Balanced CrossEntropy
            loss = F.cross_entropy(out, by, weight=model.class_weights)
            loss.backward()

            # 梯度裁剪防止小波参数爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / len(train_loader.dataset):.2f}%")

        # 3. 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                v_loss = F.cross_entropy(v_out, vy, weight=model.class_weights).item()
                val_loss += v_loss
                val_correct += (v_out.argmax(1) == vy).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_loader.dataset)

        print(f">>> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_wav_kan.pth")
            print(">>> 最佳模型已保存")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%, WAV-KAN 总参数量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_wav_kan()