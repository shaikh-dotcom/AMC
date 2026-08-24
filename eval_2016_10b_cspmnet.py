import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from CSPMNet import CSPMNet

# Configuration
BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================
# Load RadioML2016.10b & Class Mapping
# ============================================================

with open("RML2016.10b.dat", "rb") as f:
    data_10b = pickle.load(f, encoding="latin1")

split = torch.load("amc_data_split.pt", weights_only=False)

mods_train = split["mods"]
mod_to_idx = {m: i for i, m in enumerate(mods_train)}
amssb_idx = mod_to_idx["AM-SSB"]

mods_10b = sorted(set(k[0] for k in data_10b.keys()))
dropped = [m for m in mods_10b if m not in mod_to_idx]

print(f"Dropped (not in training set): {dropped}\n")


# ============================================================
# Build Evaluation Tensors (Includes IQ Shape Correction)
# ============================================================

X_list = []
y_list = []
snr_list = []

for (mod, snr), signals in data_10b.items():
    if mod not in mod_to_idx:
        continue

    for sig in signals:
        sig_arr = np.array(sig, dtype=np.float32)

        # Transpose (128, 2) to (2, 128) if necessary for the PyTorch model
        if sig_arr.shape == (128, 2):
            sig_arr = sig_arr.T

        X_list.append(sig_arr)
        y_list.append(mod_to_idx[mod])
        snr_list.append(snr)


X_10b = torch.tensor(np.array(X_list), dtype=torch.float32)
y_10b = torch.tensor(y_list, dtype=torch.long)
snr_10b = np.array(snr_list)

print(f"Total 2016.10b eval samples: {X_10b.shape[0]} (Tensor shape: {list(X_10b.shape)})\n")


# ============================================================
# Load CSPMNet Model
# ============================================================

model = CSPMNet(input_dim=[2, 128], output_dim=len(mods_train)).to(device)

model.load_state_dict(
    torch.load("cspmnet_faithful_best.pt", weights_only=True)
)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"CSPMNet parameters: {params:,}")
print("Loaded: cspmnet_faithful_best.pt\n")


# ============================================================
# DataLoader & Inference Loop
# ============================================================

loader = DataLoader(
    TensorDataset(X_10b, y_10b),
    batch_size=BATCH_SIZE,
    shuffle=False
)

correct_u = 0
correct_r = 0
total = 0

snr_correct_r = {}
snr_total = {}

with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # CSPMNet Forward Pass (returns tuple: features, logits)
        _, logits = model(X_batch)

        # Unrestricted Predictions
        preds_u = logits.argmax(dim=1)

        # Restricted Predictions (AM-SSB Masked)
        logits_masked = logits.clone()
        logits_masked[:, amssb_idx] = float("-inf")
        preds_r = logits_masked.argmax(dim=1)

        # Global Accuracy Counts
        correct_u += (preds_u == y_batch).sum().item()
        correct_r += (preds_r == y_batch).sum().item()
        total += y_batch.size(0)

        # Dynamic Per-SNR Tracking
        start = i * BATCH_SIZE
        batch_snrs = snr_10b[start : start + y_batch.size(0)]

        for j, snr in enumerate(batch_snrs):
            snr = int(snr)
            snr_total[snr] = snr_total.get(snr, 0) + 1
            if preds_r[j].item() == y_batch[j].item():
                snr_correct_r[snr] = snr_correct_r.get(snr, 0) + 1


# ============================================================
# Final Summary Output
# ============================================================

acc_u = 100.0 * correct_u / total
acc_r = 100.0 * correct_r / total
gap = acc_r - acc_u

print("============================================================")
print("CSPMNet 2016.10b TEST")
print("============================================================")
print(f"Unrestricted accuracy:               {acc_u:.2f}%")
print(f"Restricted (AM-SSB masked) accuracy: {acc_r:.2f}%")
print(f"AM-SSB attractor gap:                {gap:.2f} points")

print("\nRestricted accuracy per SNR:")
for snr in sorted(snr_total.keys()):
    acc = 100.0 * snr_correct_r.get(snr, 0) / snr_total[snr]
    print(f"SNR {snr:4d} dB: {acc:.2f}%")

print("\n--- For comparison ---")
print(
    f"CSPMNet: "
    f"restricted {acc_r:.2f}%, "
    f"in-domain clean 62.98% "
    f"(gap {62.98 - acc_r:.2f})"
)