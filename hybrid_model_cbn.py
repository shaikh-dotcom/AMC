import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexBatchNorm1d(nn.Module):
    """2x2-covariance whitening complex BatchNorm, ported from ULCNN's
    complexnn/bn.py. Jointly normalizes the (real, imag) pair per channel
    instead of normalizing each independently -- the hypothesis being
    tested here. Operates on separate zr, zi tensors of shape (B, S, T)."""
    def __init__(self, num_channels, eps=1e-4, momentum=0.1):
        super().__init__()
        C = num_channels
        self.C = C
        self.eps = eps
        self.momentum = momentum
        init_val = 1.0 / (2 ** 0.5)
        self.gamma_rr = nn.Parameter(torch.full((C,), init_val))
        self.gamma_ii = nn.Parameter(torch.full((C,), init_val))
        self.gamma_ri = nn.Parameter(torch.zeros(C))
        self.beta_r = nn.Parameter(torch.zeros(C))
        self.beta_i = nn.Parameter(torch.zeros(C))
        self.register_buffer('running_mean_r', torch.zeros(C))
        self.register_buffer('running_mean_i', torch.zeros(C))
        self.register_buffer('running_Vrr', torch.full((C,), init_val))
        self.register_buffer('running_Vii', torch.full((C,), init_val))
        self.register_buffer('running_Vri', torch.zeros(C))

    def forward(self, xr, xi):
        if self.training:
            mu_r = xr.mean(dim=(0, 2))
            mu_i = xi.mean(dim=(0, 2))
            xr_c = xr - mu_r[None, :, None]
            xi_c = xi - mu_i[None, :, None]
            Vrr = (xr_c ** 2).mean(dim=(0, 2)) + self.eps
            Vii = (xi_c ** 2).mean(dim=(0, 2)) + self.eps
            Vri = (xr_c * xi_c).mean(dim=(0, 2))
            with torch.no_grad():
                m = self.momentum
                self.running_mean_r.mul_(1 - m).add_(m * mu_r)
                self.running_mean_i.mul_(1 - m).add_(m * mu_i)
                self.running_Vrr.mul_(1 - m).add_(m * Vrr)
                self.running_Vii.mul_(1 - m).add_(m * Vii)
                self.running_Vri.mul_(1 - m).add_(m * Vri)
        else:
            mu_r, mu_i = self.running_mean_r, self.running_mean_i
            Vrr, Vii, Vri = self.running_Vrr, self.running_Vii, self.running_Vri
            xr_c = xr - mu_r[None, :, None]
            xi_c = xi - mu_i[None, :, None]

        tau = Vrr + Vii
        delta = Vrr * Vii - Vri ** 2
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inv_st = 1.0 / (s * t)
        Wrr = ((Vii + s) * inv_st)[None, :, None]
        Wii = ((Vrr + s) * inv_st)[None, :, None]
        Wri = (-Vri * inv_st)[None, :, None]

        xr_n = Wrr * xr_c + Wri * xi_c
        xi_n = Wri * xr_c + Wii * xi_c

        g_rr = self.gamma_rr[None, :, None]
        g_ii = self.gamma_ii[None, :, None]
        g_ri = self.gamma_ri[None, :, None]
        b_r = self.beta_r[None, :, None]
        b_i = self.beta_i[None, :, None]

        out_r = g_rr * xr_n + g_ri * xi_n + b_r
        out_i = g_ri * xr_n + g_ii * xi_n + b_i
        return out_r, out_i


class ComplexSubbandPhaseMotion(nn.Module):
    """Same as your original front-end, but with proper complex batchnorm
    applied to zr/zi right after the complex convolution -- replacing the
    plain real BatchNorm1d that used to sit later on the concatenated
    feature vector. This is the one change under test."""
    def __init__(self, num_subbands=8, kernel_size=15, lags=(1, 2, 4, 8)):
        super().__init__()
        self.lags = lags
        pad = kernel_size // 2
        self.conv_r = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)
        self.conv_i = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)
        self.complex_bn = ComplexBatchNorm1d(num_subbands)

    def forward(self, x):
        xr, xi = x[:, 0:1, :], x[:, 1:2, :]
        zr = self.conv_r(xr) - self.conv_i(xi)
        zi = self.conv_i(xr) + self.conv_r(xi)

        zr, zi = self.complex_bn(zr, zi)  # <-- the change under test

        eps = 1e-6
        mag = torch.sqrt(zr ** 2 + zi ** 2 + eps)
        feats = [torch.cat([torch.log1p(mag), zr, zi], dim=1)]
        for l in self.lags:
            zr_s = F.pad(zr, (l, 0))[:, :, :-l]
            zi_s = F.pad(zi, (l, 0))[:, :, :-l]
            dr = zr * zr_s + zi * zi_s
            di = zi * zr_s - zr * zi_s
            mag_d = torch.sqrt(dr ** 2 + di ** 2 + eps)
            feats.append(torch.cat([torch.log1p(mag_d), dr, di], dim=1))
        return torch.cat(feats, dim=1)


class DSCResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.dw1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dw2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw2 = nn.Conv1d(channels, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.pw1(self.dw1(x))))
        out = self.bn2(self.pw2(self.dw2(out)))
        return out + x


class DSCResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.fuse = nn.Conv1d(in_channels, out_channels, 1)
        self.unit1 = DSCResidualUnit(out_channels, kernel_size)
        self.unit2 = DSCResidualUnit(out_channels, kernel_size)
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
        return self.relu(self.conv(x)).squeeze(-1)


class HybridAMCNet(nn.Module):
    def __init__(self, num_classes=11, num_subbands=8, stem_channels=32, input_length=128):
        super().__init__()
        self.front_end = ComplexSubbandPhaseMotion(num_subbands=num_subbands)
        front_channels = 15 * num_subbands
        # No more real front_bn here -- ComplexBatchNorm1d inside the
        # front-end now handles this job, at the source, jointly on I/Q.
        self.stem = nn.Conv1d(front_channels, stem_channels, kernel_size=1)
        self.stack1 = DSCResidualStack(stem_channels, 32)
        self.stack2 = DSCResidualStack(32, 32)
        self.stack3 = DSCResidualStack(32, 32)
        self.gdwconv = GDWConv(32, input_length // 8)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.front_end(x)
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        return self.fc(self.gdwconv(x))