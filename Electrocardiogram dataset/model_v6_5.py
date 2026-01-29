import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from database_mit import prepare_ecg_data


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
    def __init__(self, in_features=187, in_hidden=32, out_hidden=32):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        # 权重矩阵 W2 负责特征投影
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.Tensor(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        nn.init.xavier_uniform_(self.W2)
        nn.init.normal_(self.b, std=1e-2)

    def forward(self, x, W1=None):
        # 1. 计算当前层的投影特征 (不含 b)，供下一层构造 KQ 或当前层构造 V
        x_norm = self.ln(x)
        proj_x = torch.matmul(x_norm, self.W2)
        base_mapping = proj_x + self.b

        if W1 is not None and torch.abs(self.beta) > 1e-6:
            # 执行注意力重组：(β*W1 + I) * base_mapping
            identity = torch.eye(self.in_features, device=x.device).unsqueeze(0)
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * identity
            out = torch.bmm(attn_kernel, base_mapping)
        else:
            out = base_mapping

        # 残差连接：仅在维度一致时生效
        final_out = out + self.gamma * x if out.shape == x.shape else out
        return final_out, proj_x


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=32, spline_order=3, hidden_dim=32, neuron_out_dim=32):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.base_scales = nn.Parameter(torch.empty(in_neurons, out_neurons))
        nn.init.normal_(self.base_scales, mean=0.6, std=0.1)

        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_hidden=hidden_dim, out_hidden=neuron_out_dim) for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.3)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_W1 = None
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.1))

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
        # x_in: [Batch, In_N, 187, 32]
        x_energy_in = torch.mean(x_in, dim=-1)
        x_norm = torch.clamp(x_energy_in / (torch.abs(x_energy_in).max(dim=-1, keepdim=True)[0] + 1e-8), -0.99, 0.99)

        # 计算样条分支与基础分支
        base_feat = F.silu(x_norm).unsqueeze(2) * self.base_scales.unsqueeze(-1).unsqueeze(0)
        basis = self.b_spline_cubic(x_norm)
        spline_feat = torch.einsum('bifc,ijc->bijf', basis, self.spline_weights)

        # 1. 构造边级激活响应
        edge_response = (base_feat + spline_feat).unsqueeze(-1)

        # 2. 应用独立边残差逻辑 (Omiga Matrix)
        # 将 omiga [In_N, Out_N] 扩展为 [1, In_N, Out_N, 1, 1] 以适配广播计算
        omiga_expanded = torch.abs(self.omiga).view(1, self.in_neurons, self.out_neurons, 1, 1)

        # 3. 最终信号 = (变换响应 + 独立残差强度) * 原始输入
        # 这确保了每一条边 (i->j) 都有自己独特的原始信号保留比例
        activated_signals = (edge_response + omiga_expanded) * x_in.unsqueeze(2)

        next_outputs = []
        next_projections = []
        t_val = torch.abs(self.temperature) + 1e-4

        # 每次前向传播前重置截获状态，防止误读旧数据
        self.last_W1 = None

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]  # [B, In_N, 187, 32]
            edge_energies = torch.mean(torch.abs(current_edges), dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)

            # 软门控控制 (Soft-Gating)
            mask = torch.sigmoid((edge_energies - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            W1_j = None
            if proj_x_prev is not None:
                # 能量/阈值比例作为权重分配
                multiplier = (edge_energies / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask

                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, i, :, :] for i in range(mid, mid * 2)], dim=-1)

                K = F.layer_norm(K, [K.size(-1)])
                Q = F.layer_norm(Q, [Q.size(-1)])

                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            # --- 核心修复：状态截获逻辑 ---
            # 如果当前神经元 j 是可视化函数正在审计的索引，则将其 W1_j 截获并保存
            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            # 聚合与计算
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


class MultiScalePrepConv(nn.Module):
    def __init__(self):
        super(MultiScalePrepConv, self).__init__()
        # 三个并行分支，捕捉不同频率特征
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2), # 抓高频细节
            nn.BatchNorm1d(8),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=15, padding=7), # 抓标准形态
            nn.BatchNorm1d(20),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=27, padding=13), # 抓长程节律(R-R间期)
            nn.BatchNorm1d(4),
            nn.SiLU()
        )
        # self.se = SEBlock(32) # 对 8+16+8=32 通道进行统一权重重分配

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # 拼接维度：[B, 32, 187]
        out = torch.cat([x1, x2, x3], dim=1)
        return out # self.se(out)


class ECGKANModel(nn.Module):
    def __init__(self, grid_size=32, spline_order=3):
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv()
        # Layer 1 & 2: 保持 32 维特征
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=32, neuron_out_dim=32)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=32, neuron_out_dim=64)
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64)
        self.layer4 = FastKANLayer(8, 5, grid_size, spline_order, hidden_dim=64, neuron_out_dim=1)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.prep_conv(x)  # [B, 32, 187]

        # 转换为 KAN 格式: [Batch, 1神经元, 187时间步, 32特征]
        x = x.transpose(1, 2).unsqueeze(1)

        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)  # x: [B, 5, 187, 1]

        # 输出层：对 1 维特征求均值(保持不变)，再对时间 187 求平均
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


def visualize_kan_weights_v6(model, sample_x, sample_y=None, layer_name='layer2', neuron_idx=0):
    """
    针对 V6.0 动态路由架构设计的可视化函数（矩阵残差更新版）：
    1. 自动处理 layer3 (32x1) 权重矩阵的 3D 渲染
    2. 真实展示由软路由生成的 W1 矩阵
    3. 整合 Beta, Alpha, Gamma, 以及新增的独立边残差 Omiga 的审计对比
    """
    model.eval()
    device = next(model.parameters()).device

    label_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

    # 1. 输入预处理 (Batch=1)
    if sample_x.dim() == 1:
        sample_x = sample_x.unsqueeze(0)
    sample_x = sample_x.to(device)

    with torch.no_grad():
        target_layer = getattr(model, layer_name)
        # 显式设置要抓取的神经元索引
        target_layer.visualize_idx = neuron_idx

        # 执行前向推理以触发状态截获钩子
        _ = model(sample_x)

        weight_layer = target_layer.weight_layers[neuron_idx]

        # 提取核心参数
        W1 = target_layer.last_W1  # [B, 187, 187]
        W2_orig = weight_layer.W2.cpu().numpy()  # [32, Out_Dim]
        b = weight_layer.b.cpu().numpy()  # [187, Out_Dim]

        # 动力学参数
        beta = torch.abs(weight_layer.beta).item()
        alpha = torch.abs(weight_layer.alpha).item()
        gamma = weight_layer.gamma.item()

        # --- 新增：提取当前神经元所关联的平均边残差 (Omiga) ---
        # target_layer.omiga 是 [In_Neurons, Out_Neurons] 矩阵
        # 我们取指向当前输出神经元 neuron_idx 的所有输入的平均残差强度
        omiga_val = torch.abs(target_layer.omiga[:, neuron_idx]).mean().item()

        # 处理 W1 绘图数据 (移除保底单位阵逻辑)
        if W1 is not None:
            W1_np = W1[0].cpu().numpy()
        else:
            # 如果是 Layer 1 或逻辑未触发，显示全黑代表无跨层注意力
            W1_np = np.zeros((187, 187))
    type_str = f" | Type: {label_map[int(sample_y)]}" if sample_y is not None else ""
    # --- 绘图逻辑开始 ---
    fig = plt.figure(figsize=(24, 12))

    # 子图 1: 原生特征变换矩阵 W2 (兼容 layer3 的 32x1 维度)
    ax1 = fig.add_subplot(231, projection='3d')
    in_dim, out_dim = W2_orig.shape

    if out_dim == 1:
        W2_plot = np.tile(W2_orig, (1, 32))
        plot_out_dim = 32
        ax1.set_xlabel('Replicated Feature (1->32)')
    else:
        W2_plot = W2_orig
        plot_out_dim = out_dim
        ax1.set_xlabel('Output Feature Dim')

    X2, Y2 = np.meshgrid(np.arange(plot_out_dim), np.arange(in_dim))
    ax1.plot_surface(X2, Y2, W2_plot, cmap='viridis', antialiased=True)
    ax1.set_title(f"1. Feature Mapping W2 ({in_dim}x{out_dim})\nNative Structure")
    ax1.set_ylabel('Input Feature Dim')

    # 子图 2: 偏置矩阵 b (Heatmap)
    ax2 = fig.add_subplot(232)
    sns.heatmap(b, cmap='magma', ax=ax2, cbar=True)
    ax2.set_title(f"2. Time-Feature Bias b (187x{out_dim})")
    ax2.set_xlabel('Feature Dim')
    ax2.set_ylabel('Time step')

    # 子图 3: 动态注意力 W1 (反映时序对齐的竖线特征)
    ax3 = fig.add_subplot(233)
    sns.heatmap(W1_np, cmap='rocket', ax=ax3, vmin=np.min(W1_np), vmax=np.max(W1_np))
    ax3.set_title(f"3. Dynamic Attention W1 (187x187)\n(Beta={beta:.3f}, Alpha={alpha:.1e})")

    # 子图 4: W1 的局部切片 (观察 R 波引起的竖线激活)
    ax4 = fig.add_subplot(234)
    row_idx = 93
    max_val = np.max(W1_np)
    if max_val > 0:
        ax4.plot(W1_np[row_idx], color='red', label=f'Row {row_idx} Activation')
        ax4.fill_between(range(187), W1_np[row_idx], alpha=0.3, color='red')
        ax4.set_ylim(0, max_val * 1.1)
        ax4.set_title(f"4. W1 Cross-section (Row {row_idx})\nInternal Focus Distribution")
    else:
        ax4.plot([0, 186], [0, 0], color='gray', linestyle='--', label='No Activation')
        ax4.set_ylim(-0.1, 1.0)
        ax4.set_title(f"4. W1 Cross-section (Row {row_idx})\n[Layer 1: No Cross-layer Attention]")
    ax4.set_xlabel("Time Step (Key)")
    ax4.set_ylabel("Attention Weight")
    ax4.legend()

    # 子图 5: 核心动力学组件分配 (Beta vs Alpha vs Gamma vs Omiga)
    ax5 = fig.add_subplot(235)
    # 增加 Omiga (紫色柱子) 代表边级残差贡献
    ax5.bar(['Attn (Beta)', 'Id (Alpha)', 'Res (Gamma)', 'Edge Res (Omiga)'],
            [beta, alpha, gamma, omiga_val],
            color=['red', 'gray', 'orange', 'purple'])
    ax5.set_title(f"5. Component Contribution (Audit)")
    ax5.set_yscale('log')  # Alpha 通常极小，建议保留对数刻度

    # 子图 6: 投影注意力 (W1 * b) -> 展现最终的时空聚焦形态
    ax6 = fig.add_subplot(236)
    W_concept = W1_np @ b
    sns.heatmap(W_concept, cmap='viridis', ax=ax6)
    ax6.set_title(f"6. Projected Attention (W1 * b)\nSpatio-Temporal Focus")

    plt.suptitle(f"Model V6.0 Topology Audit | Layer: {layer_name} | Neuron: {neuron_idx}{type_str}",
                 fontsize=18, fontweight='bold', color='darkblue')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_fast_kan_activation_v6(model, layer_name='layer1', in_idx=0, out_idx=0):
    model.eval()
    layer = getattr(model, layer_name)
    device = next(layer.parameters()).device

    # 模拟 187x32 的输入，但主要控制 187 时间轴上的变化
    x_test = torch.linspace(-1.1, 1.1, 500).to(device)
    # 模拟均值能量输入
    x_norm = x_test.view(1, 1, 500)

    with torch.no_grad():
        # SiLU 基础路径
        b_scale = layer.base_scales[in_idx, out_idx]
        base_out = b_scale * F.silu(x_norm[0, 0, :])

        # B-样条路径
        basis = layer.b_spline_cubic(x_norm) # [1, 1, 500, coeffs]
        w_spline = layer.spline_weights[in_idx, out_idx, :]
        spline_out = torch.mv(basis[0, 0, :, :], w_spline)

        y = base_out + spline_out

    plt.figure(figsize=(10, 6))
    plt.plot(x_test.cpu().numpy(), y.cpu().numpy(), label='Total Activation', lw=2.5)
    plt.plot(x_test.cpu().numpy(), base_out.cpu().numpy(), '--', label='Base (SiLU)', alpha=0.6)
    plt.plot(x_test.cpu().numpy(), spline_out.cpu().numpy(), '--', label='Spline Part', alpha=0.6)
    plt.scatter(layer.grid.cpu().numpy(), [0]*len(layer.grid), marker='|', color='red', label='Knots')
    plt.title(f"V6 Activation Edge: {layer_name} [{in_idx} -> {out_idx}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_fast_kan_topology_v6(model, layer_name='layer1'):
    """
    针对 V6.0 矩阵残差架构升级的拓扑审计函数：
    新增第 4 子图，展示每条边独立的 Omiga (Edge Residual) 权重分布。
    """
    model.eval()
    layer = getattr(model, layer_name)

    with torch.no_grad():
        # 1. 计算连接强度 (样条权重均值 + 基础缩放)
        strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.base_scales)
        strength = strength.cpu().numpy()

        # 2. 提取当前学习到的阈值 Tau
        tau = torch.abs(layer.tau).cpu().numpy()

        # 3. 计算二者差异 (差异 > 0 代表路径导通/激活)
        gap = strength - tau

        # 4. 提取独立的边残差矩阵 Omiga
        # 这一部分反映了每条边对原始信号的“物理渗透”强度
        omiga = torch.abs(layer.omiga).cpu().numpy()

    # 将画布扩展为 1x4 布局
    fig, axes = plt.subplots(1, 4, figsize=(32, 7))

    # 子图 1: 变换强度
    sns.heatmap(strength, annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[0])
    axes[0].set_title(f"1. Connection Strength\n(Weights + Scales)")

    # 子图 2: 自适应阈值
    sns.heatmap(tau, annot=True, fmt=".2f", cmap='Reds', ax=axes[1])
    axes[1].set_title(f"2. Learnable Thresholds (Tau)\n(Adaptive Gates)")

    # 子图 3: 净连通度 (决定信号是否被 Mask 截断)
    sns.heatmap(gap, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=axes[2])
    axes[2].set_title(f"3. Net Connectivity (Strength - Tau)\n(Active > 0)")

    # 子图 4: 独立边残差强度 (新增)
    # 颜色使用 Purples 以区分于其他热力图，数值越大代表原始信号保留越多
    sns.heatmap(omiga, annot=True, fmt=".3f", cmap='Purples', ax=axes[3])
    axes[3].set_title(f"4. Independent Edge Residual (Omiga)\n(Shortcut Strength)")

    plt.suptitle(f"Topology Audit: {layer_name} | V6.0 Matrix Residual Mode", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 测试运行 ---
if __name__ == "__main__":
    # 1. 实例化 grid_size=16 的模型
    # 这里的参数必须和产生那个 .pth 文件时完全一致
    model_32 = ECGKANModel(grid_size=32, spline_order=3)

    # 2. 加载对应的权重文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 请确保文件名是你之前保存 16 网格模型的文件名
    # 设置测试集文件路径
    test_csv_path = 'E:/database/archive/mitbih_test.csv'
    # 获取 val_loader
    # batch_size 建议与训练时保持一致（如 64）
    val_loader = prepare_ecg_data(test_csv_path, batch_size=64)

    state_dict = torch.load("best_kan_v6_dynamic_5.pth", map_location=device)
    model_32.load_state_dict(state_dict, strict=False)
    model_32.to(device)
    print("成功加载 grid_size=16 的模型权重")

    # 3. 调用你之前的可视化函数
    # ... 加载模型代码 ...
    if "model_32" in locals():
        # 1. 查看第一层第 0 个神经元的 187x187 融合矩阵
        # 1. 拿一个样本（比如来自测试集）
        sample_data, sample_label = next(iter(val_loader))
        test_idx = 10
        single_sample = sample_data[test_idx]  # 取 Batch 中的第一个心拍
        single_label = sample_label[test_idx].item()

        idx = 2
        # 2. 调用修改后的函数
        visualize_kan_weights_v6(model_32, single_sample, sample_y=single_label, layer_name='layer4', neuron_idx=idx)

        # 示例：查看 layer2 中，第 0 个输入神经元 到 第 7 个输出神经元 的连边激活情况
        plot_fast_kan_activation_v6(model_32, layer_name='layer3', in_idx=idx, out_idx=7)

    total_params = sum(p.numel() for p in model_32.parameters() if p.requires_grad)
    print(f"可学习参数总量 (控制点): {total_params}")

    visualize_fast_kan_topology_v6(model_32, layer_name='layer3')