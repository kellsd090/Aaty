import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from database_mit import prepare_ecg_data
from model import seed_everything


# --- 优化后的高效 KAN 层：矩阵化并行计算 ---
class FastKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=32, spline_order=3):
        super(FastKANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.num_coeffs = grid_size + spline_order

        # 3D 权重张量并行化 [in, out, coeffs]
        self.spline_weight = nn.Parameter(torch.randn(in_dim, out_dim, self.num_coeffs) * 0.1)
        self.base_scale = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))

    def b_spline_cubic(self, x):
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res = torch.zeros_like(dist)
        mask1 = dist < 1.0
        res[mask1] = (2 / 3) - (dist[mask1] ** 2) + (dist[mask1] ** 3 / 2)
        mask2 = (dist >= 1.0) & (dist < 2.0)
        res[mask2] = ((2 - dist[mask2]) ** 3) / 6
        return res

    def forward(self, x):
        x_norm = torch.clamp(x, -0.99, 0.99)
        # 基础路径 (Base Linear-like)
        base_act = F.silu(x_norm).unsqueeze(-1)
        base_out = (base_act * self.base_scale).sum(dim=1)
        # 样条路径 (Spline Nonlinear)
        basis = self.b_spline_cubic(x_norm)
        spline_out = torch.einsum('bic,ijc->bj', basis, self.spline_weight)
        return base_out + spline_out


class OriginalECGKAN(nn.Module):
    def __init__(self, grid_size=32):
        super(OriginalECGKAN, self).__init__()
        # 结构相似性对标: 输入(187) -> 隐藏1(32) -> 隐藏2(32) -> 输出(5)
        # 虽然数字不是8，但层级结构完全对标你的 1-8-8-5 逻辑
        self.kan1 = FastKANLayer(187, 32, grid_size)
        self.kan2 = FastKANLayer(32, 32, grid_size)
        self.kan3 = FastKANLayer(32, 5, grid_size)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        out = self.kan3(x)
        return out


# --- 训练逻辑补全 ---
def train_original_kan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    # 实例化更强大的 KAN
    model = OriginalECGKAN(grid_size=32).to(device)

    # 类别权重统计（含平方根平滑）
    all_labels = []
    for _, y in train_loader: all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)])
    model.class_weights = torch.sqrt(weights / weights.mean()).to(device)

    # ================= [初始状态评估补全] =================
    model.eval()
    init_loss = 0
    init_correct = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            init_loss += F.cross_entropy(v_out, vy, weight=model.class_weights).item()
            init_correct += (v_out.argmax(1) == vy).sum().item()

    avg_init_loss = init_loss / len(val_loader)
    print(f"\n[Initial State] Loss: {avg_init_loss:.6f}, Val Acc: {100 * init_correct / len(val_loader.dataset):.2f}%")
    print("-" * 75)

    # 优化配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_acc = 0

    # 训练循环
    for epoch in range(50):  # 增加 Epoch 以适应更复杂的 KAN
        model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"KAN Epoch {epoch + 1}/50")

        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out, by, weight=model.class_weights)
            loss.backward()

            # 梯度裁剪防止样条发散
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / len(train_loader.dataset):.2f}%")

        # 验证阶段补全
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
            torch.save(model.state_dict(), "best_original_kan.pth")
            print(">>> 最佳 KAN 模型已保存")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%, KAN 参数总量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_original_kan()