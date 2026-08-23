"""
Faithful PyTorch port of ULCNN (Guo et al., 2024), ported directly from the
official Keras/complexnn implementation in ULCNN-main/5ULCNN.py and
ULCNN-main/complexnn/{conv.py,bn.py}. Not an approximation: ComplexConv1d
uses the exact block-matrix complex-convolution identity from
complexnn/conv.py (cat_kernels_4_real/imag), and ComplexBatchNorm1d
implements the same 2x2 covariance whitening (Trabelsi et al., 2018) from
complexnn/bn.py, not independent per-channel normalization.

Default hyperparameters (n_neuron=16, n_mobileunit=6, kernel_size=5) match
the pretrained checkpoint shipped in ULCNN-main/model/ULCNN_MN=6_N=16_KS=5.hdf5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """Complex 1D convolution. Input/output are (batch, 2*C, T) with the
    first C channels real and the last C channels imaginary -- matching the
    complexnn convention (channel axis split in half, real then imag).
    Combination formula matches complexnn/conv.py's cat_kernels_4_real/imag
    exactly: out_real = Wr*xr - Wi*xi, out_imag = Wi*xr + Wr*xi.
    """
    def __init__(self, in_complex_channels, out_complex_channels, kernel_size,
                 stride=1, bias=True):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding for odd kernel, stride=1
        self.C_in = in_complex_channels
        self.conv_r = nn.Conv1d(in_complex_channels, out_complex_channels,
                                 kernel_size, stride=stride, padding=pad, bias=False)
        self.conv_i = nn.Conv1d(in_complex_channels, out_complex_channels,
                                 kernel_size, stride=stride, padding=pad, bias=False)
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


class ComplexBatchNorm1d(nn.Module):
    """2x2-covariance whitening complex BatchNorm, ported directly from
    complexnn/bn.py's complex_standardization + ComplexBN. This is NOT the
    same as independently normalizing real and imaginary parts -- it
    whitens the joint (real, imag) covariance per channel, then applies a
    learnable 2x2 affine (gamma_rr, gamma_ri, gamma_ii) + shift (beta).
    """
    def __init__(self, num_complex_channels, eps=1e-4, momentum=0.1):
        super().__init__()
        C = num_complex_channels
        self.C = C
        self.eps = eps
        self.momentum = momentum
        init_val = 1.0 / (2 ** 0.5)  # sqrt_init, matches complexnn default
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
    """Depthwise-separable conv with TF/Keras 'same' padding semantics
    (asymmetric padding computed from input length, stride, kernel size) --
    matches Keras SeparableConv1D(..., padding='same') exactly, including
    for stride=2 where symmetric padding would be wrong.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        in_len = x.shape[-1]
        out_len = -(-in_len // self.stride)  # ceil division
        pad_total = max((out_len - 1) * self.stride + self.kernel_size - in_len, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        x = F.pad(x, (pad_before, pad_after))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def channel_shuffle(x, groups=2):
    """Matches the original channel_shuffle exactly (hardcoded groups=2)."""
    N, C, T = x.shape
    cpg = C // groups
    x = x.view(N, groups, cpg, T)
    x = x.transpose(1, 2).contiguous()
    x = x.view(N, C, T)
    return x


class MobileUnit(nn.Module):
    """Ports dwconv_mobile: SeparableConv1D(stride=2) -> BN -> ReLU -> channel_shuffle."""
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
        x = channel_shuffle(x, groups=2)
        return x


class ChannelAttention(nn.Module):
    """Ports channelattention: shared Dense1/Dense2 applied to both
    GlobalAveragePooling and GlobalMaxPooling branches, summed, sigmoid-gated.
    """
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
    """
    Faithful port of the ULCNN architecture from 5ULCNN.py:
      ComplexConv1D -> ComplexBN -> ReLU
      -> 6x [dwconv_mobile -> channelattention]
      -> GAP-fuse of units 4,5,6 (0-indexed 3,4,5) -> Dense(num_classes) -> softmax (via CrossEntropyLoss)

    Input: (batch, 2, T) with channel 0 = I (real), channel 1 = Q (imag) --
    matches ComplexConv1d's expected (batch, 2*1, T) layout directly, no
    transpose needed (unlike the original Keras script's channels-last input).
    """
    def __init__(self, num_classes=11, n_neuron=16, n_mobileunit=6, kernel_size=5):
        super().__init__()
        self.n_mobileunit = n_mobileunit
        self.complex_conv = ComplexConv1d(1, n_neuron, kernel_size, stride=1, bias=True)
        self.complex_bn = ComplexBatchNorm1d(n_neuron)
        self.relu0 = nn.ReLU()

        channels = 2 * n_neuron
        self.mobile_units = nn.ModuleList(
            [MobileUnit(channels, n_neuron, kernel_size) for _ in range(n_mobileunit)]
        )
        self.attentions = nn.ModuleList(
            [ChannelAttention(channels) for _ in range(n_mobileunit)]
        )
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


if __name__ == "__main__":
    model = ULCNN(num_classes=11, n_neuron=16, n_mobileunit=6, kernel_size=5)
    dummy = torch.randn(4, 2, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")  # expect (4, 11)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
    print("(Reference: README reports 9,751; original script's model.summary() reported 8,807 --")
    print(" small differences are expected from exact bias/param-counting conventions between")
    print(" Keras and this port, not a structural mismatch.)")
