import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexSubbandPhaseMotion(nn.Module):
    def __init__(self, num_subbands=8, kernel_size=15, lags=(1, 2, 4, 8)):
        super().__init__()
        self.S = num_subbands
        self.lags = lags
        pad = kernel_size // 2
        self.conv_r = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)
        self.conv_i = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        xr = x[:, 0:1, :]
        xi = x[:, 1:2, :]
        zr = self.conv_r(xr) - self.conv_i(xi)
        zi = self.conv_i(xr) + self.conv_r(xi)
        eps = 1e-6
        mag = torch.sqrt(zr ** 2 + zi ** 2 + eps)
        b = torch.cat([torch.log1p(mag), zr, zi], dim=1)
        features = [b]
        for l in self.lags:
            zr_shift = F.pad(zr, (l, 0))[:, :, :-l]
            zi_shift = F.pad(zi, (l, 0))[:, :, :-l]
            delta_r = zr * zr_shift + zi * zi_shift
            delta_i = zi * zr_shift - zr * zi_shift
            mag_d = torch.sqrt(delta_r ** 2 + delta_i ** 2 + eps)
            d_l = torch.cat([torch.log1p(mag_d), delta_r, delta_i], dim=1)
            features.append(d_l)
        return torch.cat(features, dim=1)


class DSCResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=5, use_batchnorm=False):
        super().__init__()
        pad = kernel_size // 2
        self.use_bn = use_batchnorm
        self.depthwise1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pointwise1 = nn.Conv1d(channels, channels, 1)
        self.depthwise2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pointwise2 = nn.Conv1d(channels, channels, 1)
        self.relu = nn.ReLU()
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(channels)
            self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.pointwise1(self.depthwise1(x))
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise2(self.depthwise2(out))
        if self.use_bn:
            out = self.bn2(out)
        return out + residual


class DSCResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, use_batchnorm=False):
        super().__init__()
        self.fuse = nn.Conv1d(in_channels, out_channels, 1)
        self.unit1 = DSCResidualUnit(out_channels, kernel_size, use_batchnorm)
        self.unit2 = DSCResidualUnit(out_channels, kernel_size, use_batchnorm)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.fuse(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return self.pool(x)


class GDWConv(nn.Module):
    def __init__(self, channels, time_length):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=time_length, groups=channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out.squeeze(-1)


class HybridAMCNet(nn.Module):
    def __init__(self, num_classes=11, num_subbands=8, input_length=128, use_batchnorm=False):
        super().__init__()
        self.use_bn = use_batchnorm
        self.front_end = ComplexSubbandPhaseMotion(num_subbands=num_subbands)
        front_end_channels = 15 * num_subbands

        if use_batchnorm:
            self.front_bn = nn.BatchNorm1d(front_end_channels)

        self.stem = nn.Conv1d(front_end_channels, 64, kernel_size=1)
        self.stack1 = DSCResidualStack(64, 32, use_batchnorm=use_batchnorm)
        self.stack2 = DSCResidualStack(32, 32, use_batchnorm=use_batchnorm)
        self.stack3 = DSCResidualStack(32, 32, use_batchnorm=use_batchnorm)

        final_time_length = input_length // (2 ** 3)
        self.gdwconv = GDWConv(32, final_time_length)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.front_end(x)
        if self.use_bn:
            x = self.front_bn(x)
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.gdwconv(x)
        return self.fc(x)