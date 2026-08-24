import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ulcnn_model import ULCNN


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================
# Load RadioML2016.10b
# ============================================================

with open("RML2016.10b.dat", "rb") as f:
    data_10b = pickle.load(f, encoding="latin1")


# ============================================================
# Load training split / class mapping
# ============================================================

split = torch.load(
    "amc_data_split.pt",
    weights_only=False
)

mods_train = split["mods"]

mod_to_idx = {
    m: i for i, m in enumerate(mods_train)
}

# AM-SSB class index in the training class mapping
amssb_idx = mod_to_idx["AM-SSB"]


# ============================================================
# Check available classes in 2016.10b
# ============================================================

mods_10b = sorted(
    set(k[0] for k in data_10b.keys())
)

dropped = [
    m for m in mods_10b
    if m not in mod_to_idx
]

print(f"Dropped (not in training set): {dropped}\n")


# ============================================================
# Build evaluation dataset
# ============================================================

X_list = []
y_list = []
snr_list = []

for (mod, snr), signals in data_10b.items():

    if mod not in mod_to_idx:
        continue

    for sig in signals:

        X_list.append(sig)
        y_list.append(mod_to_idx[mod])
        snr_list.append(snr)


X_10b = torch.tensor(
    np.array(X_list),
    dtype=torch.float32
)

y_10b = torch.tensor(
    y_list,
    dtype=torch.long
)

snr_10b = np.array(snr_list)


print(
    f"Total 2016.10b eval samples: "
    f"{X_10b.shape[0]} "
    f"(Tensor shape: {list(X_10b.shape)})\n"
)


# ============================================================
# Load faithful ULCNN
# ============================================================

model = ULCNN(
    num_classes=len(mods_train),
    n_neuron=16,
    n_mobileunit=6,
    kernel_size=5
).to(device)


model.load_state_dict(
    torch.load(
        "ulcnn_faithful_best.pt",
        weights_only=True
    )
)

model.eval()


# Count parameters
params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"ULCNN (faithful port) parameters: "
    f"{params:,}"
)

print("Loaded: ulcnn_faithful_best.pt\n")


# ============================================================
# DataLoader
# ============================================================

loader = DataLoader(
    TensorDataset(X_10b, y_10b),
    batch_size=256,
    shuffle=False
)


# ============================================================
# Evaluation
# ============================================================

correct_u = 0
correct_r = 0
total = 0

snr_correct_r = {}
snr_total = {}


with torch.no_grad():

    for i, (X_batch, y_batch) in enumerate(loader):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(X_batch)

        # ----------------------------------------------------
        # Unrestricted prediction
        # ----------------------------------------------------

        preds_u = logits.argmax(dim=1)

        # ----------------------------------------------------
        # Restricted prediction
        # AM-SSB cannot be selected
        # ----------------------------------------------------

        logits_masked = logits.clone()

        logits_masked[:, amssb_idx] = float("-inf")

        preds_r = logits_masked.argmax(dim=1)

        # ----------------------------------------------------
        # Overall accuracy
        # ----------------------------------------------------

        correct_u += (
            preds_u == y_batch
        ).sum().item()

        correct_r += (
            preds_r == y_batch
        ).sum().item()

        total += y_batch.size(0)

        # ----------------------------------------------------
        # Per-SNR restricted accuracy
        # ----------------------------------------------------

        start = i * 256

        batch_snrs = snr_10b[
            start:start + y_batch.size(0)
        ]

        for j, snr in enumerate(batch_snrs):

            snr = int(snr)

            snr_total[snr] = (
                snr_total.get(snr, 0) + 1
            )

            if preds_r[j].item() == y_batch[j].item():

                snr_correct_r[snr] = (
                    snr_correct_r.get(snr, 0) + 1
                )


# ============================================================
# Final results
# ============================================================

acc_u = 100.0 * correct_u / total
acc_r = 100.0 * correct_r / total

gap = acc_r - acc_u


print("============================================================")
print("ULCNN 2016.10b TEST")
print("============================================================")

print(
    f"Unrestricted accuracy:               "
    f"{acc_u:.2f}%"
)

print(
    f"Restricted (AM-SSB masked) accuracy: "
    f"{acc_r:.2f}%"
)

print(
    f"AM-SSB attractor gap:                "
    f"{gap:.2f} points"
)


# ============================================================
# Restricted accuracy per SNR
# ============================================================

print("\nRestricted accuracy per SNR:")

for snr in sorted(snr_total.keys()):

    acc = (
        100.0
        * snr_correct_r.get(snr, 0)
        / snr_total[snr]
    )

    print(
        f"SNR {snr:4d} dB: {acc:.2f}%"
    )


# ============================================================
# Comparison with in-domain result
# ============================================================

IN_DOMAIN_CLEAN = 60.13

print("\n--- For comparison ---")

print(
    f"ULCNN: restricted {acc_r:.2f}%, "
    f"in-domain clean {IN_DOMAIN_CLEAN:.2f}% "
    f"(gap {IN_DOMAIN_CLEAN - acc_r:.2f})"
)