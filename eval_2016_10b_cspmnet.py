import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from CSPMNet import CSPMNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

with open('RML2016.10b.dat', 'rb') as f:  # match your actual filename
    data_10b = pickle.load(f, encoding='latin1')

split = torch.load('amc_data_split.pt', weights_only=False)
mods_train = split['mods']
mod_to_idx = {m: i for i, m in enumerate(mods_train)}
amssb_idx = mod_to_idx['AM-SSB']

mods_10b = sorted(set(k[0] for k in data_10b.keys()))
dropped = [m for m in mods_10b if m not in mod_to_idx]
print(f"Dropped (not in training set): {dropped}\n")

X_list, y_list, snr_list = [], [], []
for (mod, snr), signals in data_10b.items():
    if mod not in mod_to_idx:
        continue
    for sig in signals:
        X_list.append(sig)
        y_list.append(mod_to_idx[mod])
        snr_list.append(snr)

X_10b = torch.tensor(np.array(X_list), dtype=torch.float32)
y_10b = torch.tensor(y_list, dtype=torch.long)
snr_10b = np.array(snr_list)
print(f"Total 2016.10b eval samples: {X_10b.shape[0]}\n")

# Fixed: real constructor signature is (input_dim, output_dim), not num_classes --
# confirmed from eval_domain_shift_cspmnet.py, the script that already works.
model = CSPMNet(input_dim=[2, 128], output_dim=len(mods_train)).to(device)
model.load_state_dict(torch.load('cspmnet_faithful_best.pt', weights_only=True))
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"CSPMNet parameters: {params:,}")
print("Loaded: cspmnet_faithful_best.pt\n")

loader = DataLoader(TensorDataset(X_10b, y_10b), batch_size=256, shuffle=False)

correct_u, correct_r, total = 0, 0, 0
snr_correct_r, snr_total = {}, {}

with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Fixed: forward returns a tuple, matching eval_domain_shift_cspmnet.py
        _, logits = model(X_batch)

        preds_u = logits.argmax(dim=1)

        logits_masked = logits.clone()
        logits_masked[:, amssb_idx] = float('-inf')
        preds_r = logits_masked.argmax(dim=1)

        correct_u += (preds_u == y_batch).sum().item()
        correct_r += (preds_r == y_batch).sum().item()
        total += y_batch.size(0)

        # per-SNR (restricted) tracking, batch-aligned
        start = i * 256
        batch_snrs = snr_10b[start:start + y_batch.size(0)]
        for j, snr in enumerate(batch_snrs):
            snr = int(snr)
            snr_total[snr] = snr_total.get(snr, 0) + 1
            if preds_r[j].item() == y_batch[j].item():
                snr_correct_r[snr] = snr_correct_r.get(snr, 0) + 1

acc_u = 100 * correct_u / total
acc_r = 100 * correct_r / total
print(f"CSPMNet unrestricted accuracy: {acc_u:.2f}%")
print(f"CSPMNet restricted (AM-SSB masked) accuracy: {acc_r:.2f}%")
print(f"AM-SSB attractor gap: {acc_r - acc_u:.2f} points\n")

print("Restricted accuracy per SNR:")
for snr in sorted(snr_total.keys()):
    acc = 100 * snr_correct_r.get(snr, 0) / snr_total[snr]
    print(f"SNR {snr:4d} dB: {acc:.2f}%")

print(f"\n--- For comparison ---")
print(f"HybridAMCNet+ComplexBN: restricted 61.02%, in-domain clean 61.33% (gap 0.31)")
print(f"CSPMNet:                restricted {acc_r:.2f}%, in-domain clean 62.98% (gap {62.98-acc_r:.2f})")