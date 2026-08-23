import numpy as np
import torch

torch.manual_seed(0)

# ---------------------------------------------------------
# Load original test data
# ---------------------------------------------------------
data = torch.load("amc_data_split.pt", weights_only=False)

X_test = data["X_test"]       # (N, 2, 128)
y_test = data["y_test"]
snr_test = data["snr_test"]
mods = data["mods"]


# ---------------------------------------------------------
# Rician / Rayleigh block fading
# ---------------------------------------------------------
def rician_block_fade(X, K):
    N = X.shape[0]

    theta = torch.rand(N) * 2 * np.pi

    los_r = np.sqrt(K / (K + 1)) * torch.cos(theta)
    los_i = np.sqrt(K / (K + 1)) * torch.sin(theta)

    scat_r = torch.randn(N) * np.sqrt(0.5 / (K + 1))
    scat_i = torch.randn(N) * np.sqrt(0.5 / (K + 1))

    h_r = (los_r + scat_r).view(N, 1)
    h_i = (los_i + scat_i).view(N, 1)

    I = X[:, 0, :]
    Q = X[:, 1, :]

    I_out = h_r * I - h_i * Q
    Q_out = h_r * Q + h_i * I

    return torch.stack([I_out, Q_out], dim=1)


# ---------------------------------------------------------
# Carrier Frequency Offset
# ---------------------------------------------------------
def add_cfo(X, max_norm_freq=0.01):

    N, _, L = X.shape

    f = (torch.rand(N, 1) * 2 - 1) * max_norm_freq

    n = torch.arange(L, dtype=torch.float32).view(1, L)

    phase = 2 * np.pi * f * n

    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)

    I = X[:, 0, :]
    Q = X[:, 1, :]

    I_out = cos_p * I - sin_p * Q
    Q_out = sin_p * I + cos_p * Q

    return torch.stack([I_out, Q_out], dim=1)


# ---------------------------------------------------------
# Generate domain-shift conditions
# ---------------------------------------------------------
conditions = {
    "clean": X_test,

    "rician_K10": rician_block_fade(
        X_test, K=10
    ),

    "rician_K3": rician_block_fade(
        X_test, K=3
    ),

    "rician_K1": rician_block_fade(
        X_test, K=1
    ),

    "rayleigh_K0": rician_block_fade(
        X_test, K=0
    ),

    "cfo": add_cfo(
        X_test,
        max_norm_freq=0.01
    ),
}


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
torch.save(
    {
        "conditions": conditions,
        "y_test": y_test,
        "snr_test": snr_test,
        "mods": mods,
    },
    "shifted_test_sets.pt"
)


print("\nSaved shifted_test_sets.pt")
print("Conditions:")

for name, X in conditions.items():
    print(
        f"  {name:12s} "
        f"shape={tuple(X.shape)} "
        f"mean|X|={X.abs().mean():.4f}"
    )

print("\nDone.")