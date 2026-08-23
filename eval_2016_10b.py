import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from hybrid_model_cbn import HybridAMCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('RML2016.10b.dat', 'rb') as f:  # match your filename
    data_10b = pickle.load(f, encoding='latin1')

split = torch.load('amc_data_split.pt', weights_only=False)
mods_train = split['mods']
mod_to_idx = {m: i for i, m in enumerate(mods_train)}
amssb_idx = mod_to_idx['AM-SSB']  # the structurally-irrelevant class here

mods_10b = sorted(set(k[0] for k in data_10b.keys()))
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

model = HybridAMCNet(num_classes=len(mods_train), num_subbands=8,
                      stem_channels=32, input_length=128).to(device)
model.load_state_dict(torch.load('hybrid_cbn_final.pt', weights_only=True))
model.eval()

loader = DataLoader(TensorDataset(X_10b, y_10b), batch_size=256, shuffle=False)

correct_unrestricted, correct_restricted, total = 0, 0, 0
snr_correct_u, snr_correct_r, snr_total = {}, {}, {}

with torch.no_grad():
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)

        preds_unrestricted = logits.argmax(dim=1)

        logits_masked = logits.clone()
        logits_masked[:, amssb_idx] = float('-inf')  # AM-SSB is not a valid choice here
        preds_restricted = logits_masked.argmax(dim=1)

        correct_unrestricted += (preds_unrestricted == y_batch).sum().item()
        correct_restricted += (preds_restricted == y_batch).sum().item()
        total += y_batch.size(0)

acc_u = 100 * correct_unrestricted / total
acc_r = 100 * correct_restricted / total
print(f"Unrestricted (as-deployed) accuracy: {acc_u:.2f}%")
print(f"Restricted (AM-SSB masked, shared-class-only) accuracy: {acc_r:.2f}%")
print(f"Gap caused purely by the AM-SSB attractor: {acc_r - acc_u:.2f} points")