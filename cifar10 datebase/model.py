import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from dataset_cifar10 import get_dataloader, add_coordinate_encoding


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplineWeightLayer(nn.Module):
    def __init__(self, in_features=256, in_hidden=64, out_hidden=64, kernel_size=3):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 256 (16x16)
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size

        # 特征投影矩阵 W2
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        # 偏置 b 改为初始为 0
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        # 空间权重控制参数
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # I 的幅度
        self.beta = nn.Parameter(torch.ones(1))  # W1 的幅度
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # W3 的幅度 (初始较小)
        self.theta = nn.Parameter(torch.ones(1))  # 残差幅度

        # --- [核心修改] W3 局部卷积先验 ---
        # groups=out_hidden 实现了你要求的“全部维度同时进行独立二维卷积”
        # padding=k//2 自动处理了你说的“16x16 -> 18x18 补0”逻辑
        self.W3_conv = nn.Conv2d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False
        )

        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()

        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        # x: [B, 256, Hidden]
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)

        # 1. 计算 x' = LN(x)*W2 + b
        # 这里对应你设想的 (LN(x)*W2 + b)
        x_prime = torch.matmul(x_norm, self.W2) + self.b  # [B, 256, out_hidden]

        # 2. 计算 W3(x')：局部空间先验
        # 恢复 16x16 空间进行卷积
        x_spatial = x_prime.transpose(1, 2).view(b, self.out_hidden, 16, 16)
        w3_feat = self.W3_conv(x_spatial)  # 卷积自动处理了 padding 和滑窗
        w3_feat = w3_feat.view(b, self.out_hidden, seq_len).transpose(1, 2)

        # 3. 空间组合：(β*W1 + α*I) * x' + θ*W3(x')
        identity = torch.eye(seq_len, device=x.device).unsqueeze(0)

        # 注意：W3 在你的公式里是作为独立加项还是在算子内部？
        # 按照你的公式：((β*W1 + α*I + θ*W3) * x')
        # 我们这里通过元素加法实现等效逻辑
        if W1 is not None:
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat

        # 4. 加上残差 γ*res(x)
        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64, kernel_size=3):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.base_scales = nn.Parameter(torch.empty(in_neurons, out_neurons))
        nn.init.normal_(self.base_scales, mean=0.6, std=0.1)

        # 每一个输出神经元对应一个增强版的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size) for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_W1 = None
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))

    # b_spline_cubic 等方法保持不变...
    def b_spline_cubic(self, x):
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res = torch.zeros_like(dist)
        mask1 = dist < 1.0
        res[mask1] = (2 / 3) - (dist[mask1] ** 2) + (dist[mask1] ** 3 / 2)
        mask2 = (dist >= 1.0) & (dist < 2.0)
        res[mask2] = ((2 - dist[mask2]) ** 3) / 6
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape
        # 计算样条输入
        x_energy_in = torch.mean(x_in, dim=-1)
        x_norm = torch.clamp(x_energy_in / (torch.abs(x_energy_in).max(dim=-1, keepdim=True)[0] + 1e-8), -0.99, 0.99)

        base_feat = F.silu(x_norm).unsqueeze(2) * self.base_scales.unsqueeze(-1).unsqueeze(0)
        basis = self.b_spline_cubic(x_norm)
        spline_feat = torch.einsum('bifc,ijc->bijf', basis, self.spline_weights)

        edge_response = (base_feat + spline_feat).unsqueeze(-1)
        omiga_expanded = torch.abs(self.omiga).view(1, self.in_neurons, self.out_neurons, 1, 1)
        activated_signals = (edge_response + omiga_expanded) * x_in.unsqueeze(2)

        next_outputs, next_projections = [], []
        t_val = torch.abs(self.temperature) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(torch.abs(current_edges), dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((edge_energies - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            W1_j = None
            if proj_x_prev is not None:
                multiplier = (edge_energies / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, i, :, :] for i in range(mid, mid * 2)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            combined_input = torch.mean(current_edges * mask, dim=1)
            out_j, proj_j = self.weight_layers[j](combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)

        return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ImagePrepConv(nn.Module):
    def __init__(self, in_channels=5, out_channels=32):
        super(ImagePrepConv, self).__init__()

        # --- 1. 维度投影分支 (Shortcut) ---
        # 用于将 5 通道输入直接映射到 out_channels 维度，以便执行加法
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # --- 2. 残差学习分支 (Residual Branches) ---
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

        # --- 3. 融合后的进一步提取层 ---
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 激活函数
        self.relu = nn.SiLU()
        # 降采样：将 32x32 缩小为 16x16
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, 5, 32, 32]
        # 计算残差路径
        identity = self.shortcut(x)  # [B, 64, 32, 32]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # 拼接多尺度特征
        out = torch.cat([x1, x2, x3], dim=1)  # [B, 64, 32, 32]
        # 进一步卷积增强
        out = self.post_conv(out)
        # --- 核心：执行残差加法 ---
        out += identity
        out = self.relu(out)
        # 降采样并转为 KAN 序列格式
        #out = self.pool(out)  # [B, 64, 16, 16]
        out = self.pixel_unshuffle(out)
        b, c, h, w = out.shape
        out = out.flatten(2)  # [B, 64, 256]
        out = out.transpose(1, 2)  # [B, 256, 64]
        out = out.unsqueeze(1)  # [B, 1, 256, 64]
        return out


class ECGKANModel(nn.Module):
    def __init__(self, grid_size=64, spline_order=3):
        super(ECGKANModel, self).__init__()
        # 1. 初始多尺度卷积，将 5x32x32 映射为序列 [B, 1, 256, 64]
        # 注意这里我显式设置了 out_channels=64
        self.prep_conv = ImagePrepConv(in_channels=5, out_channels=32)

        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=128, neuron_out_dim=64, kernel_size=7)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64, kernel_size=5)
        self.layer3 = FastKANLayer(8, 16, grid_size, spline_order, hidden_dim=64, neuron_out_dim=128, kernel_size=5)
        self.layer4 = FastKANLayer(16, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer5 = FastKANLayer(16, 20, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer6 = FastKANLayer(20, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=16, kernel_size=3)
        self.layer7 = FastKANLayer(16, 10, grid_size, spline_order, hidden_dim=16, neuron_out_dim=1, kernel_size=1)

    def forward(self, x):
        # x: [B, 5, 32, 32]
        # 修正：ImagePrepConv 内部已经处理好了维度展开
        x = self.prep_conv(x)  # 输出应为 [B, 1, 256, 64]

        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)
        x, proj5 = self.layer5(x, proj_x_prev=proj4)
        x, proj6 = self.layer6(x, proj_x_prev=proj5)
        x, proj7 = self.layer7(x, proj_x_prev=proj6)

        # x: [B, 10, 256, 1] -> [B, 10]
        # 对 256 个像素序列点求平均，得到最终分类概率
        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        # 保持原有审计逻辑不变
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.base_scales)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {"active": active_count, "total": m.in_neurons * m.out_neurons,
                                  "tau_mean": torch.abs(m.tau).mean().item(), 'ratio': active_count / (m.in_neurons * m.out_neurons)}
        return info


def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    weight_layer = layer.weight_layers[neuron_idx]

    # 扩展画布为 3行2列，以容纳 W3
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- (1) W2 矩阵热力图 ---
    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")

    # --- (2) W1 矩阵热力图 (动态生成) ---
    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')

    # --- (3) [新增] W3 矩阵可视化 (局部卷积先验) ---
    # 逻辑：将 k*k 卷积核展开为 1D 稀疏向量 [k^2 + 2*(16-k)个0]
    with torch.no_grad():
        # 提取 Depthwise 卷积核: [Hidden, 1, k, k]
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k, _ = conv_weight.shape

        # 构造稀疏展示矩阵: d * [k + (16-k) + k + (16-k) + k]
        # 总长度 = 3k + 2(16-k) = k + 32
        sparse_w3 = []
        for d in range(d_out):
            kernel_2d = conv_weight[d, 0]  # [k, k]
            row_list = []
            for r in range(k):
                row_list.append(kernel_2d[r])  # 加入 k 个权重
                if r < k - 1:
                    row_list.append(torch.zeros(16 - k))  # 插入 16-k 个 0
            sparse_w3.append(torch.cat(row_list))

        w3_display = torch.stack(sparse_w3).numpy()
        sns.heatmap(w3_display, ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 Sparse Matrix (Local Prior)")
        axes[1, 0].set_xlabel("Sparse Kernel Index (3k + 2*(16-k))")
        axes[1, 0].set_ylabel("Channel Dimension (d)")

    # --- (4) θ, α, β, γ 参数柱状图 (权重分配比例) ---
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()
    }
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")
    axes[1, 1].set_ylim(0, max(params.values()) * 1.2)

    # --- (5) SiLU + B样条激活形状 ---
    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)
        base_scale = layer.base_scales[edge_idx[0], edge_idx[1]]
        y_base = base_scale * F.silu(x_range)

        # 兼容方法名：支持 b_spl_cubic 或 b_spline_cubic
        spl_func = layer.b_spline_cubic if hasattr(layer, 'b_spline_cubic') else layer.b_spl_cubic
        basis = spl_func(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()

        axes[2, 0].plot(x_range.cpu(), y_base.cpu(), '--', label='Base (SiLU)', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), (y_base + y_spline).cpu(), 'r', lw=2, label='Combined')
        axes[2, 0].set_title(f"Activation: Edge ({edge_idx[0]} -> {edge_idx[1]})")
        axes[2, 0].legend();
        axes[2, 0].grid(True)

    # --- (6) 层间导通热力图 ---
    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.base_scales)
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], annot=False, cmap='Blues', cbar=False)
        axes[2, 1].set_title(f"Layer {layer_idx}: Edge Conductivity")
        axes[2, 1].set_xlabel("Output Neurons");
        axes[2, 1].set_ylabel("Input Neurons")

    plt.show()


# --- 测试运行 ---
# --- 修改后的主函数：加载本地权重并可视化 ---
if __name__ == "__main__":
    # 1. 设置路径（根据你图中的文件名和 Colab 的实际路径修改）
    # 假设你已经把文件上传到了 Colab 根目录，或者挂载了 Drive
    checkpoint_path = 'D:\\treasure\\true\\thesis\\pth\\cifar10\\layer7\\best_model_5th_9111.pth'

    # 2. 实例化模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGKANModel().to(device)

    if os.path.exists(checkpoint_path):
        # 加载权重，map_location 增加兼容性
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载权重文件: {checkpoint_path}")
    else:
        print(f"错误: 未在当前目录找到 {checkpoint_path}")

    # 3. 输出模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- KAN 模型参数统计 ---")
    print(f"总参数量: {total_params:,}")
    print(f"----------------------\n")

    # 4. 准备可视化数据
    # 必须提供 5 通道输入以匹配 ImagePrepConv
    dummy_input = torch.randn(1, 5, 32, 32).to(device)

    # 审计第 3 层 (特征最丰富的阶段)
    target_layer_idx = 5
    target_neuron_idx = 10
    target_layer = getattr(model, f'layer{target_layer_idx}')
    target_layer.visualize_idx = target_neuron_idx

    # 运行一次前向传播以截获动态矩阵 W1
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)


    # 5. 核心修复：手动修正函数内的名称错误并调用
    # 直接在调用前修复函数内部的引用错误
    def fixed_visualize(model, layer_idx, neuron_idx):
        # 临时覆盖函数内的错误调用名
        layer = getattr(model, f'layer{layer_idx}')
        # 确保 visualize_model_internals 里的 layer.b_spl_cubic 指向正确的 b_spline_cubic
        if not hasattr(layer, 'b_spl_cubic'):
            layer.b_spl_cubic = layer.b_spline_cubic
        visualize_model_internals(model, layer_idx=layer_idx, neuron_idx=neuron_idx)


    print(f"正在生成第 {target_layer_idx} 层可视化图表...")
    fixed_visualize(model, layer_idx=target_layer_idx, neuron_idx=0)