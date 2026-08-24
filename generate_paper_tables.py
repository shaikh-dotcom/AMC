import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

### =====================================================================
### CHECKPOINT PATHS -- edit these to match your actual filenames
### =====================================================================
CKPT = {
    "original_8_64":     "ablation_narrowband_aug_and_bn.pt",
    "backend_cut_8_32":   "lever_final_backend_cut_full.pt",
    "frontend_cut_4_64":  "lever_final_frontend_cut_full.pt",
    "reduced_4_32":       "ablation_narrowband_reduced_aug_and_bn.pt",
    "plain_8_64":         "ablation_narrowband_plain.pt",
    "aug_only_8_64":      "ablation_narrowband_aug_only.pt",
    "bn_only_8_64":       "ablation_narrowband_bn_only.pt",
    "rola_net":           "hybrid_cbn_final.pt",
    "ulcnn":              "ulcnn_faithful_best.pt",
    "cspmnet":            "cspmnet_faithful_best.pt",
}
DATA_SPLIT = "amc_data_split.pt"
SHIFTED_SETS = "shifted_test_sets.pt"  # optional -- skip Rician/CFO columns if missing

def exists(path):
    ok = os.path.isfile(path)
    if not ok:
        print(f"  [SKIP] {path} not found.")
    return ok

### =====================================================================
### MODEL DEFINITIONS -- fixed and updated to avoid loading mismatches
### =====================================================================

class ComplexSubbandPhaseMotion(nn.Module):
    def __init__(self, num_subbands=8, kernel_size=15, lags=(1, 2, 4, 8)):
        super().__init__()
        self.lags = lags
        pad = kernel_size // 2
        self.conv_r = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)
        self.conv_i = nn.Conv1d(1, num_subbands, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        xr, xi = x[:, 0:1, :], x[:, 1:2, :]
        zr = self.conv_r(xr) - self.conv_i(xi)
        zi = self.conv_i(xr) + self.conv_r(xi)
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

# --- Standard units used for HybridAMCNet_PlainBN and RoLANet ---
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

# --- Hybrid model definition with real BatchNorm ---
class HybridAMCNet_PlainBN(nn.Module):
    """Matches param_lever_final.py and backend_cut_final.py -- always-on real BatchNorm."""
    def __init__(self, num_classes=11, num_subbands=8, stem_channels=64, input_length=128):
        super().__init__()
        self.front_end = ComplexSubbandPhaseMotion(num_subbands=num_subbands)
        front_channels = 15 * num_subbands
        self.front_bn = nn.BatchNorm1d(front_channels)
        self.stem = nn.Conv1d(front_channels, stem_channels, kernel_size=1)
        self.stack1 = DSCResidualStack(stem_channels, 32)
        self.stack2 = DSCResidualStack(32, 32)
        self.stack3 = DSCResidualStack(32, 32)
        self.gdwconv = GDWConv(32, input_length // 8)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.front_end(x)
        x = self.front_bn(x)
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        return self.fc(self.gdwconv(x))


# --- FIXES for Toggle models: layer names changed to depthwise/pointwise to match state_dict ---

class DSCResidualUnitToggle(nn.Module):
    """Fixed: Uses depthwise1/pointwise1 layer names instead of dw1/pw1 to align with original state_dict keys."""
    def __init__(self, channels, kernel_size=5, use_batchnorm=True):
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

class DSCResidualStackToggle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, use_batchnorm=True):
        super().__init__()
        self.fuse = nn.Conv1d(in_channels, out_channels, 1)
        self.unit1 = DSCResidualUnitToggle(out_channels, kernel_size, use_batchnorm)
        self.unit2 = DSCResidualUnitToggle(out_channels, kernel_size, use_batchnorm)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.fuse(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return self.pool(x)

class HybridAMCNet_Toggle(nn.Module):
    """Fixed: Added front_bn layer and depthwise/pointwise residual units to match saved checkpoints."""
    def __init__(self, num_classes=11, num_subbands=8, input_length=128, use_batchnorm=True):
        super().__init__()
        self.use_bn = use_batchnorm
        self.front_end = ComplexSubbandPhaseMotion(num_subbands=num_subbands)
        front_channels = 15 * num_subbands
        if use_batchnorm:
            self.front_bn = nn.BatchNorm1d(front_channels)
        self.stem = nn.Conv1d(front_channels, 64, kernel_size=1)
        self.stack1 = DSCResidualStackToggle(64, 32, use_batchnorm=use_batchnorm)
        self.stack2 = DSCResidualStackToggle(32, 32, use_batchnorm=use_batchnorm)
        self.stack3 = DSCResidualStackToggle(32, 32, use_batchnorm=use_batchnorm)
        self.gdwconv = GDWConv(32, input_length // 8)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.front_end(x)
        if self.use_bn:
            x = self.front_bn(x)
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        return self.fc(self.gdwconv(x))

class HybridAMCNet_Toggle_Reduced(HybridAMCNet_Toggle):
    """Fixed: Uses depthwise/pointwise layers to match the reduced 4-subband checkpoint exactly."""
    def __init__(self, num_classes=11, num_subbands=4, input_length=128, use_batchnorm=True):
        nn.Module.__init__(self)
        self.use_bn = use_batchnorm
        self.front_end = ComplexSubbandPhaseMotion(num_subbands=num_subbands)
        front_channels = 15 * num_subbands
        if use_batchnorm:
            self.front_bn = nn.BatchNorm1d(front_channels)
        self.stem = nn.Conv1d(front_channels, 32, kernel_size=1)
        self.stack1 = DSCResidualStackToggle(32, 32, use_batchnorm=use_batchnorm)
        self.stack2 = DSCResidualStackToggle(32, 32, use_batchnorm=use_batchnorm)
        self.stack3 = DSCResidualStackToggle(32, 32, use_batchnorm=use_batchnorm)
        self.gdwconv = GDWConv(32, input_length // 8)
        self.fc = nn.Linear(32, num_classes)


# --- RoLA-Net with joint Complex BatchNorm (CBN) ---

class ComplexBatchNorm1d(nn.Module):
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

class ComplexSubbandPhaseMotion_CBN(nn.Module):
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
        zr, zi = self.complex_bn(zr, zi)
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

class RoLANet(nn.Module):
    """Matches hybrid_model_cbn.py exactly. This is RoLA-Net."""
    def __init__(self, num_classes=11, num_subbands=8, stem_channels=32, input_length=128):
        super().__init__()
        self.front_end = ComplexSubbandPhaseMotion_CBN(num_subbands=num_subbands)
        front_channels = 15 * num_subbands
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


# --- CSPMNet and subcomponents ---

def morlet_wavelet_1d(length: int, center_freq: float, sigma: float) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) - length // 2
    gaussian = torch.exp(-(t**2) / (2.0 * sigma**2))
    carrier = torch.exp(1j * 2.0 * math.pi * center_freq * t)
    psi = gaussian.to(torch.complex64) * carrier.to(torch.complex64)
    return psi / psi.abs().sum()

def build_filter_bank(J: int, Q: int, filter_len: int) -> list[torch.Tensor]:
    wavelets = []
    for j in range(1, J * Q + 1):
        scale = 2.0 ** (j / Q)
        center_freq = 0.5 / scale
        sigma = scale
        wavelets.append(morlet_wavelet_1d(filter_len, center_freq, sigma))
    return wavelets

def shift_right_with_repeat(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    if steps >= x.shape[-1]:
        raise ValueError(f"steps must be smaller than signal length, got steps={steps}")
    prefix = x[..., :1].expand(*x.shape[:-1], steps)
    return torch.cat([prefix, x[..., :-steps]], dim=-1)

class ScaledAdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.query_projection = nn.Linear(feature_dim, attention_dim)
        self.key_projection = nn.Linear(feature_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.last_attention_weights = None

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if query.dim() != 2:
            raise ValueError(f"Expected query shape (B, C), got {tuple(query.shape)}")
        if values.dim() != 3:
            raise ValueError(f"Expected values shape (B, T, C), got {tuple(values.shape)}")
        if query.shape[0] != values.shape[0] or query.shape[1] != values.shape[2]:
            raise ValueError("Query and values must share batch and feature dimensions")
        projected_query = self.query_projection(query).unsqueeze(1)
        projected_keys = self.key_projection(values)
        scores = self.score_projection(torch.tanh(projected_query + projected_keys)).squeeze(-1)
        scores = scores / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(scores, dim=1)
        self.last_attention_weights = attention_weights.detach()
        attention_weights = self.dropout(attention_weights)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        return self.layer_norm(context)

class TemporalAttentionFusionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, mixer_channels: int = 128, rnn_hidden: int = 96, feature_dim: int = 64, dense_hidden: int = 96) -> None:
        super().__init__()
        self.mixer_channels = mixer_channels
        self.rnn_hidden = rnn_hidden
        self.feature_dim = feature_dim
        self.rnn_feature_dim = 2 * rnn_hidden
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels, mixer_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mixer_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.rnn = nn.GRU(input_size=mixer_channels, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.rnn_norm = nn.BatchNorm1d(self.rnn_feature_dim)
        self.rnn_dropout = nn.Dropout(0.3)
        self.attention = ScaledAdditiveAttention(feature_dim=self.rnn_feature_dim, attention_dim=rnn_hidden, dropout=0.1)
        self.fc1 = nn.Linear(self.rnn_feature_dim, dense_hidden)
        self.fc_norm1 = nn.BatchNorm1d(dense_hidden)
        self.fc_dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dense_hidden, feature_dim)
        self.fc_norm2 = nn.BatchNorm1d(feature_dim)
        self.fc_dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _batch_norm_sequence(x: torch.Tensor, batch_norm: nn.BatchNorm1d) -> torch.Tensor:
        x = batch_norm(x.transpose(1, 2))
        return x.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mixer(x)
        x = x.transpose(1, 2).contiguous()
        x, _ = self.rnn(x)
        x = self._batch_norm_sequence(x, self.rnn_norm)
        x = self.rnn_dropout(x)
        query = x[:, -1, :]
        x = self.attention(query, x)
        x = F.relu(self.fc_norm1(self.fc1(x)))
        x = self.fc_dropout1(x)
        feature = F.relu(self.fc_norm2(self.fc2(x)))
        feature = self.fc_dropout2(feature)
        logits = self.fc_out(feature)
        return feature, logits

class FreeConvSubbandFrontEnd(nn.Module):
    def __init__(self, J: int = 2, Q: int = 4, filter_len: int = 33) -> None:
        super().__init__()
        self.J = int(J)
        self.Q = int(Q)
        self.filter_len = int(filter_len)
        wavelets = build_filter_bank(self.J, self.Q, self.filter_len)
        self.num_subbands = len(wavelets)
        weight_real = torch.stack([wavelet.real for wavelet in wavelets], dim=0).unsqueeze(1)
        weight_imag = torch.stack([wavelet.imag for wavelet in wavelets], dim=0).unsqueeze(1)
        self.weight_real = nn.Parameter(weight_real)
        self.weight_imag = nn.Parameter(weight_imag)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (B, 2, T), got {tuple(x.shape)}")
        pad = self.filter_len // 2
        x_real = x[:, 0:1, :]
        x_imag = x[:, 1:2, :]
        y_real = F.conv1d(x_real, self.weight_real, padding=pad) - F.conv1d(x_imag, self.weight_imag, padding=pad)
        y_imag = F.conv1d(x_real, self.weight_imag, padding=pad) + F.conv1d(x_imag, self.weight_real, padding=pad)
        signal_len = x.shape[-1]
        return y_real[..., :signal_len], y_imag[..., :signal_len]

class SubbandPhaseMotionFeatureMap(nn.Module):
    def __init__(self, lags: tuple[int, ...] = (1, 2, 4, 8), eps: float = 1e-6) -> None:
        super().__init__()
        self.lags = tuple(lags)
        self.eps = float(eps)

    def output_channels(self, num_subbands: int) -> int:
        return 3 * int(num_subbands) * (1 + len(self.lags))

    def _base_features(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> list[torch.Tensor]:
        magnitude = torch.sqrt(z_real.pow(2) + z_imag.pow(2) + self.eps)
        return [torch.log1p(magnitude), z_real, z_imag]

    def _delta_features(self, z_real: torch.Tensor, z_imag: torch.Tensor, lag: int) -> list[torch.Tensor]:
        prev_real = shift_right_with_repeat(z_real, lag)
        prev_imag = shift_right_with_repeat(z_imag, lag)
        delta_real = z_real * prev_real + z_imag * prev_imag
        delta_imag = z_imag * prev_real - z_real * prev_imag
        delta_mag = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.eps)
        return [torch.log1p(delta_mag), delta_real, delta_imag]

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> torch.Tensor:
        features = self._base_features(z_real, z_imag)
        for lag in self.lags:
            features.extend(self._delta_features(z_real, z_imag, lag))
        return torch.cat(features, dim=1)

class SubbandPhaseMotionAttnRNNBase(nn.Module):
    def __init__(self, input_dim, output_dim: int, front_end: nn.Module, lags: tuple[int, ...] = (1, 2, 4, 8), mixer_channels: int = 128, rnn_hidden: int = 96, feature_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.front_end = front_end
        self.feature_map = SubbandPhaseMotionFeatureMap(lags=lags)
        self.feature_channels = self.feature_map.output_channels(front_end.num_subbands)
        self.feature_norm = nn.BatchNorm1d(self.feature_channels)
        self.fusion = TemporalAttentionFusionHead(in_channels=self.feature_channels, num_classes=output_dim, mixer_channels=mixer_channels, rnn_hidden=rnn_hidden, feature_dim=feature_dim)

    def extract_subband_phase_motion_map(self, x: torch.Tensor) -> torch.Tensor:
        z_real, z_imag = self.front_end(x)
        features = self.feature_map(z_real, z_imag)
        return self.feature_norm(features)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        features = self.extract_subband_phase_motion_map(x)
        feature, logits = self.fusion(features)
        if return_feature:
            return feature, logits
        return None, logits

    def count_parameters(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total_trainable": trainable, "total_all": total}

class CSPMNet(SubbandPhaseMotionAttnRNNBase):
    """CSPMNet model for 2-channel I/Q automatic modulation classification."""
    def __init__(self, input_dim=[2, 128], output_dim=11, J: int = 2, Q: int = 4, filter_len: int = 33, lags: tuple[int, ...] = (1, 2, 4, 8), mixer_channels: int = 128, rnn_hidden: int = 96, feature_dim: int = 64) -> None:
        front_end = FreeConvSubbandFrontEnd(J=J, Q=Q, filter_len=filter_len)
        super().__init__(input_dim=input_dim, output_dim=output_dim, front_end=front_end, lags=lags, mixer_channels=mixer_channels, rnn_hidden=rnn_hidden, feature_dim=feature_dim)


# --- ULCNN model implementation ---

class ULCNN_ComplexConv1d(nn.Module):
    def __init__(self, in_complex_channels, out_complex_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.C_in = in_complex_channels
        self.conv_r = nn.Conv1d(in_complex_channels, out_complex_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.conv_i = nn.Conv1d(in_complex_channels, out_complex_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.use_bias = bias
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_complex_channels))
            self.bias_i = nn.Parameter(torch.zeros(out_complex_channels))

    def forward(self, x):
        C = self.C_in
        xr, xi = x[:, :C, :], x[:, C:, :]
        out_r = self.conv_r(xr) - self.conv_i(xi)
        out_i = self.conv_i(xr) + self.conv_r(xi)
        if self.use_bias:
            out_r = out_r + self.bias_r[None, :, None]
            out_i = out_i + self.bias_i[None, :, None]
        return torch.cat([out_r, out_i], dim=1)

class ULCNN_ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_complex_channels, eps=1e-4, momentum=0.1):
        super().__init__()
        C = num_complex_channels
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

    def forward(self, x):
        C = self.C
        xr, xi = x[:, :C, :], x[:, C:, :]
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
        return torch.cat([out_r, out_i], dim=1)

class SeparableConv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        in_len = x.shape[-1]
        out_len = -(-in_len // self.stride)
        pad_total = max((out_len - 1) * self.stride + self.kernel_size - in_len, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        x = F.pad(x, (pad_before, pad_after))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def channel_shuffle(x, groups=2):
    N, C, T = x.shape
    cpg = C // groups
    x = x.view(N, groups, cpg, T)
    x = x.transpose(1, 2).contiguous()
    return x.view(N, C, T)

class MobileUnit(nn.Module):
    def __init__(self, in_channels, neurons, kernel_size=5):
        super().__init__()
        out_channels = 2 * neurons
        self.sep = SeparableConv1dSame(in_channels, out_channels, kernel_size, stride=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sep(x)
        x = self.bn(x)
        x = self.relu(x)
        return channel_shuffle(x, groups=2)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = x.mean(dim=2)
        gmp = x.amax(dim=2)
        gap_att = self.fc2(self.relu(self.fc1(gap)))
        gmp_att = self.fc2(self.relu(self.fc1(gmp)))
        mask = self.sigmoid(gap_att + gmp_att)
        return x * mask.unsqueeze(-1)

class ULCNN(nn.Module):
    """Faithful PyTorch port of ULCNN (Guo et al., 2024)."""
    def __init__(self, num_classes=11, n_neuron=16, n_mobileunit=6, kernel_size=5):
        super().__init__()
        self.n_mobileunit = n_mobileunit
        self.complex_conv = ULCNN_ComplexConv1d(1, n_neuron, kernel_size, stride=1, bias=True)
        self.complex_bn = ULCNN_ComplexBatchNorm1d(n_neuron)
        self.relu0 = nn.ReLU()
        channels = 2 * n_neuron
        self.mobile_units = nn.ModuleList([MobileUnit(channels, n_neuron, kernel_size) for _ in range(n_mobileunit)])
        self.attentions = nn.ModuleList([ChannelAttention(channels) for _ in range(n_mobileunit)])
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.complex_conv(x)
        x = self.complex_bn(x)
        x = self.relu0(x)
        feats = []
        for i in range(self.n_mobileunit):
            x = self.mobile_units[i](x)
            x = self.attentions[i](x)
            if i in (3, 4, 5):
                feats.append(x.mean(dim=2))
        f = feats[0] + feats[1] + feats[2]
        return self.fc(f)


### =====================================================================
### EVALUATION FUNCTIONS
### =====================================================================

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            # handle CSPMNet outputting (feature, logits)
            out = model(Xb)
            logits = out[1] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total

def evaluate_band(model, X_test, y_test, snr_test, lo=-12, hi=-6):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(X_test)):
            snr = int(snr_test[i])
            if not (lo <= snr <= hi):
                continue
            x = X_test[i].unsqueeze(0).to(device)
            out = model(x)
            logits = out[1] if isinstance(out, tuple) else out
            pred = logits.argmax(1).item()
            total += 1
            if pred == y_test[i].item():
                correct += 1
    return 100 * correct / total if total else float('nan')

def evaluate_snr_points(model, X_test, y_test, snr_test, points):
    model.eval()
    out = {}
    with torch.no_grad():
        for snr_target in points:
            correct, total = 0, 0
            for i in range(len(X_test)):
                if int(snr_test[i]) != snr_target:
                    continue
                x = X_test[i].unsqueeze(0).to(device)
                outputs = model(x)
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                pred = logits.argmax(1).item()
                total += 1
                if pred == y_test[i].item():
                    correct += 1
            out[snr_target] = 100 * correct / total if total else float('nan')
    return out

def evaluate_condition(model, X, y_test):
    model.eval()
    loader = DataLoader(TensorDataset(X, y_test), batch_size=256, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            logits = out[1] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total

def load_model(cls, ckpt_key, **kwargs):
    path = CKPT[ckpt_key]
    if not exists(path):
        return None
    model = cls(**kwargs).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

### =====================================================================
### LATENCY BENCHMARKING
### =====================================================================

def benchmark_latency(model, device_name, n_warmup=20, n_runs=100):
    dev = torch.device(device_name)
    model = model.to(dev)
    model.eval()
    x = torch.randn(1, 2, 128).to(dev)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device_name == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return 1000 * elapsed / n_runs  # ms per single-sample inference


### =====================================================================
### MAIN EXECUTION
### =====================================================================

def main():
    if not exists(DATA_SPLIT):
        print(f"Error: {DATA_SPLIT} not found. Please place your data split in the current directory.")
        return

    split = torch.load(DATA_SPLIT, weights_only=False)
    X_test, y_test, snr_test = split['X_test'], split['y_test'], split['snr_test']
    mods = split['mods']
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    shifted = None
    if exists(SHIFTED_SETS):
        shifted = torch.load(SHIFTED_SETS, weights_only=False)

    # --- TABLE 2 ---
    print("=" * 70)
    print("TABLE 2: RoLA-Net architecture blueprint (discovered)")
    print("=" * 70)
    rola = load_model(RoLANet, "rola_net", num_classes=len(mods), num_subbands=8, stem_channels=32)
    if rola is not None:
        shapes = {}
        def make_hook(name):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                shapes[name] = tuple(o.shape)
                return hook
            return hook
        handles = [child.register_forward_hook(make_hook(name)) for name, child in rola.named_children()]
        with torch.no_grad():
            rola(torch.randn(1, 2, 128).to(device))
        for h in handles:
            h.remove()
        
        print(f"{'Layer':<15}{'Output Shape':<24}{'Params':>10}")
        total_p = 0
        for name, child in rola.named_children():
            p = sum(x.numel() for x in child.parameters())
            shape_str = str(shapes.get(name, "N/A"))
            print(f"{name:<15}{shape_str:<24}{p:>10,}")
            total_p += p
        print(f"{'TOTAL':<39}{total_p:>10,}")
    print()

    # --- TABLE 3 ---
    print("=" * 70)
    print("TABLE 3: Width configuration study (real 2x2 grid)")
    print("=" * 70)
    configs_t3 = [
        ("8 subbands / 64 stem (original)", HybridAMCNet_Toggle, "original_8_64", dict(num_classes=len(mods), num_subbands=8, use_batchnorm=True)),
        ("8 subbands / 32 stem (backend-cut)", HybridAMCNet_PlainBN, "backend_cut_8_32", dict(num_classes=len(mods), num_subbands=8, stem_channels=32)),
        ("4 subbands / 64 stem (frontend-cut)", HybridAMCNet_PlainBN, "frontend_cut_4_64", dict(num_classes=len(mods), num_subbands=4, stem_channels=64)),
        ("4 subbands / 32 stem (reduced)", HybridAMCNet_Toggle_Reduced, "reduced_4_32", dict(num_classes=len(mods), num_subbands=4, use_batchnorm=True)),
    ]
    print(f"{'Config':<38}{'Params':>9}{'TestAcc':>10}{'Band Acc':>10}")
    for name, cls, key, kwargs in configs_t3:
        m = load_model(cls, key, **kwargs)
        if m is None:
            continue
        p = sum(x.numel() for x in m.parameters())
        acc = evaluate(m, test_loader)
        band = evaluate_band(m, X_test, y_test, snr_test)
        print(f"{name:<38}{p:>9,}{acc:>9.2f}%{band:>9.2f}%")
    print()

    # --- TABLE 4 ---
    print("=" * 70)
    print("TABLE 4: Parameter, latency benchmarking (CPU + GPU only)")
    print("=" * 70)
    bench_models = []
    m = load_model(RoLANet, "rola_net", num_classes=len(mods), num_subbands=8, stem_channels=32)
    if m is not None:
        bench_models.append(("RoLA-Net (ours)", m))
    m = load_model(HybridAMCNet_PlainBN, "backend_cut_8_32", num_classes=len(mods), num_subbands=8, stem_channels=32)
    if m is not None:
        bench_models.append(("backend_cut_full", m))
    m = load_model(ULCNN, "ulcnn", num_classes=len(mods), n_neuron=16, n_mobileunit=6, kernel_size=5)
    if m is not None:
        bench_models.append(("ULCNN", m))
    m = load_model(CSPMNet, "cspmnet", input_dim=[2, 128], output_dim=len(mods))
    if m is not None:
        bench_models.append(("CSPMNet", m))

    print(f"{'Model':<20}{'Params':>9}{'CPU (ms)':>12}{'GPU (ms)':>12}")
    for name, m in bench_models:
        p = sum(x.numel() for x in m.parameters())
        cpu_ms = benchmark_latency(m, "cpu")
        gpu_ms = benchmark_latency(m, "cuda") if torch.cuda.is_available() else float('nan')
        print(f"{name:<20}{p:>9,}{cpu_ms:>11.3f}{gpu_ms:>11.3f}")
    print()

    # --- TABLE 6 ---
    print("=" * 70)
    print("TABLE 6: Augmentation/BatchNorm ablation (discovered)")
    print("=" * 70)
    configs_t6 = [
        ("8-subband / plain",     HybridAMCNet_Toggle, "plain_8_64", dict(num_classes=len(mods), num_subbands=8, use_batchnorm=False)),
        ("8-subband / aug_only",  HybridAMCNet_Toggle, "aug_only_8_64", dict(num_classes=len(mods), num_subbands=8, use_batchnorm=False)),
        ("8-subband / bn_only",   HybridAMCNet_Toggle, "bn_only_8_64", dict(num_classes=len(mods), num_subbands=8, use_batchnorm=True)),
        ("8-subband / aug_and_bn", HybridAMCNet_Toggle, "original_8_64", dict(num_classes=len(mods), num_subbands=8, use_batchnorm=True)),
    ]
    print(f"{'Config':<26}{'Params':>9}{'TestAcc':>10}")
    for name, cls, key, kwargs in configs_t6:
        m = load_model(cls, key, **kwargs)
        if m is None:
            continue
        p = sum(x.numel() for x in m.parameters())
        acc = evaluate(m, test_loader)
        print(f"{name:<26}{p:>9,}{acc:>9.2f}%")
    print()

    # --- TABLE 7 ---
    if shifted is not None:
        print("=" * 70)
        print("TABLE 7: Domain-shift robustness (discovered)")
        print("=" * 70)
        conditions = shifted["conditions"]
        models_t7 = []
        m = load_model(RoLANet, "rola_net", num_classes=len(mods), num_subbands=8, stem_channels=32)
        if m is not None:
            models_t7.append(("RoLA-Net (ours)", m))
        m = load_model(ULCNN, "ulcnn", num_classes=len(mods), n_neuron=16, n_mobileunit=6, kernel_size=5)
        if m is not None:
            models_t7.append(("ULCNN", m))
        m = load_model(CSPMNet, "cspmnet", input_dim=[2, 128], output_dim=len(mods))
        if m is not None:
            models_t7.append(("CSPMNet", m))

        print(f"{'Model':<20}" + "".join(f"{cond:>14}" for cond in conditions.keys()))
        for name, m in models_t7:
            row_str = f"{name:<20}"
            for cond_name, X_cond in conditions.items():
                acc = evaluate_condition(m, X_cond, y_test)
                row_str += f"{acc:>13.2f}%"
            print(row_str)
        print()
    else:
        print("TABLE 7 skipped: shifted_test_sets.pt not found.\n")

    # --- TABLE 8 ---
    print("=" * 70)
    print("TABLE 8: SNR crossover matrix (discovered)")
    print("=" * 70)
    snr_points = [-20, -12, -6, 0, 6, 12, 18]
    models_t8 = []
    m = load_model(RoLANet, "rola_net", num_classes=len(mods), num_subbands=8, stem_channels=32)
    if m is not None:
        models_t8.append(("RoLA-Net (ours)", m))
    m = load_model(HybridAMCNet_PlainBN, "backend_cut_8_32", num_classes=len(mods), num_subbands=8, stem_channels=32)
    if m is not None:
        models_t8.append(("backend_cut_full", m))
    m = load_model(ULCNN, "ulcnn", num_classes=len(mods), n_neuron=16, n_mobileunit=6, kernel_size=5)
    if m is not None:
        models_t8.append(("ULCNN", m))
    m = load_model(CSPMNet, "cspmnet", input_dim=[2, 128], output_dim=len(mods))
    if m is not None:
        models_t8.append(("CSPMNet", m))

    print(f"{'Model':<20}" + "".join(f"{s:>8}dB" for s in snr_points))
    for name, m in models_t8:
        accs = evaluate_snr_points(m, X_test, y_test, snr_test, snr_points)
        row = "".join(f"{accs[s]:>9.2f}%" for s in snr_points)
        print(f"{name:<20}{row}")
    print()

if __name__ == "__main__":
    main()
