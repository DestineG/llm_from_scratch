'''
normalize/
├── base.py
├── minmax.py
├── zscore.py
├── robust.py
├── lpnorm.py
├── batchnorm.py
├── layernorm.py
└── registry.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinMaxNormalizer(nn.Module):
    """
    将输入张量沿指定维度进行 Min-Max 归一化，将值缩放到 [min_val, max_val] 范围内。
    默认情况下，min_val=0.0，max_val=1.0。
    归一化公式：
        normalized_x = (x - x_min) / (x_max - x_min)
        scaled_x = normalized_x * (max_val - min_val) + min_val
        其中，x_min 和 x_max 分别是输入张量在指定维度上的最小值和最大值。
    Args:
        min_val (float): 归一化后的最小值，默认值为 0.0。
        max_val (float): 归一化后的最大值，默认值为 1.0。
    Inputs:
        x (torch.Tensor): 输入张量。
        dim (int): 指定归一化的维度，默认值为 0。
        eps (float): 防止除零的小常数，默认值为 1e-8。
    Returns:
        torch.Tensor: 归一化后的张量，值范围在 [min_val, max_val] 之间。
    """
    def __init__(self, min_val=0.0, max_val=1.0):
        super(MinMaxNormalizer, self).__init__()
        assert min_val < max_val, "min_val must be less than max_val"
        # 让 min 和 max 跟随 model 一起流浪
        self.register_buffer('min_val', torch.tensor(min_val))
        self.register_buffer('max_val', torch.tensor(max_val))
    
    def forward(self, x, dim=0, eps=1e-8):
        # dim=1: (b, c, h, w) -> (b, 1, h, w)
        x_min = x.min(dim=dim, keepdim=True).values
        # dim=1: (b, c, h, w) -> (b, 1, h, w)
        x_max = x.max(dim=dim, keepdim=True).values

        # (b, 1, h, w) -> (b, 1, h, w)[0~1]
        normalized_x = (x - x_min) / (x_max - x_min + eps)
        # (b, 1, h, w)[0~1] -> (b, c, h, w)[min_val~max_val]
        target_min = self.min_val.to(x.dtype)
        target_max = self.max_val.to(x.dtype)
        scaled_x = normalized_x * (target_max - target_min) + target_min
        return scaled_x

class ZScoreNormalizer(nn.Module):
    """
    将输入张量沿指定维度进行 Z-Score 标准化。
    标准化公式：
        zscore_x = (x - mean) / std
        其中，mean 和 std 分别是输入张量在指定维度上的均值和标准差。
    Inputs:
        x (torch.Tensor): 输入张量。
        dim (int): 指定标准化的维度，默认值为 0。
        eps (float): 防止除零的小常数，默认值为 1e-8。
    Returns:
        torch.Tensor: 标准化后的张量，均值为 0，标准差为 1。
    """
    def __init__(self):
        super(ZScoreNormalizer, self).__init__()

    def forward(self, x, dim=0, eps=1e-8):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        zscore_x = (x - mean) / (std + eps)
        return zscore_x

class RobustNormalizer(nn.Module):
    """
    将输入张量沿指定维度进行鲁棒标准化，使用中位数和四分位距 (IQR)。
    标准化公式：
        robust_x = (x - median) / IQR
        其中，median 是输入张量在指定维度上的中位数，IQR 是四分位距 (Q3 - Q1)。
    Inputs:
        x (torch.Tensor): 输入张量。
        dim (int): 指定标准化的维度，默认值为 0。
        eps (float): 防止除零的小常数，默认值为 1e-8。
    Returns:
        torch.Tensor: 标准化后的张量。
    """
    def __init__(self):
        super(RobustNormalizer, self).__init__()

    def forward(self, x, dim=0, eps=1e-8):
        Q = x.quantile(torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype, device=x.device), dim=dim, keepdim=True)
        Q1, median, Q3 = Q[0], Q[1], Q[2]
        IQR = Q3 - Q1
        robust_x = (x - median) / (IQR + eps)
        return robust_x

class LpNormalizer(nn.Module):
    """
    将输入张量沿指定维度进行 Lp 范数归一化。
    归一化公式：
        normalized_x = x / ||x||_p
        其中，||x||_p 是输入张量在指定维度上的 Lp 范数。
    Args:
        p (float): Lp 范数的阶数，必须大于等于 1 或等于无穷大 (inf)。默认值为 2。
    Inputs:
        x (torch.Tensor): 输入张量。
        dim (int): 指定归一化的维度，默认值为 0。
        eps (float): 防止除零的小常数，默认值为 1e-8。
    Returns:
        torch.Tensor: 归一化后的张量。
    """
    def __init__(self, p=2):
        super(LpNormalizer, self).__init__()
        assert p >= 1 or p == float('inf'), "p must be >= 1 or inf"
        self.p = p
    
    def forward(self, x, dim=0, eps=1e-8):
        return F.normalize(x, p=self.p, dim=dim, eps=eps)

class BatchNormalizer(nn.Module):
    def __init__(self, num_features, dim=1, eps=1e-5, momentum=0.1, affine=True):
        """
        特征维度的 Batch Normalization

        Args:
            num_features (int): 特征维度长度
            dim (int): 特征所在的轴，默认 1
            eps (float): 防止除零
            momentum (float): running mean / var 更新动量
            affine (bool): 是否启用可学习的缩放(weight)与平移(bias)
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # running statistics（不参与反向传播）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # affine parameters（可学习）
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # 所有非 feature 维度都参与统计
        reduce_dims = [d for d in range(x.ndim) if d != self.dim]

        if self.training:
            # (b, c, h, w) -> mean: (1, c, 1, 1), var: (1, c, 1, 1)
            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)

            # 更新 running statistics(不参与反向传播)
            with torch.no_grad():
                # (1, c, 1, 1) -> curr_mean: (c,), curr_var: (c,)
                curr_mean = mean.view(-1)
                curr_var = var.view(-1)

                # 更新 running statistics
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * curr_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * curr_var)
        else:
            # (1, 1, 1, 1) -> (1, c, 1, 1)
            shape = [1] * x.ndim
            shape[self.dim] = -1

            # (c,) -> (1, c, 1, 1)
            mean = self.running_mean.view(shape)
            var = self.running_var.view(shape)

        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 对每个特征仿射变换
        if self.affine:
            shape = [1] * x.ndim
            shape[self.dim] = -1
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x_hat = weight * x_hat + bias

        return x_hat

class LayerNormalizer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        """
        通用 Layer Normalization（等价 nn.LayerNorm）

        Args:
            normalized_shape (int or tuple): 要归一化的维度形状（通常是最后 K 维）
            eps (float): 防止除零
            affine (bool): 是否启用可学习参数 weight / bias
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # 对最后 len(normalized_shape) 个维度做 LN
        # 
        dims = tuple(range(-len(self.normalized_shape), 0))

        # dims=2 (b, c, h, w) -> (b, c, 1, 1)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 对每个坐标仿射变换
        if self.affine:
            # dim=2 (b, c, h, w), (h, w), (h, w) -> (b, c, h, w)
            x_hat = x_hat * self.weight + self.bias

        return x_hat
