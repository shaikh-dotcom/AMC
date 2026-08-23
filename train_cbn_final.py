import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from hybrid_model_cbn import HybridAMCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

data = torch.load('amc_data_split.pt', weights_only=False)
X_train_full, y_train_full = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
mods = data['mods']
snr_test = data['snr_test']
snr_train_full = np.asarray(data['snr_train'])

X_train_base, X_val, y_train_base, y_val, snr_train, snr_val = train_test_split(
    X_train_full.numpy(), y_train_full.numpy(), snr_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full.numpy()
)
X_train_base = torch.tensor(X_train_base, dtype=torch.float32)
y_train_base = torch.tensor(y_train_base, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
snr_val = torch.tensor(snr_val, dtype=torch.long)


def rotate_augment(X, y):
    I, Q = X[:, 0, :], X[:, 1, :]
    variants = [torch.stack([I, Q], 1), torch.stack([Q, -I], 1),
                torch.stack([-I, -Q], 1), torch.stack([-Q, I], 1)]
    return torch.cat(variants, 0), y.repeat(4)


X_train, y_train = rotate_augment(X_train_base, y_train_base)
print(f"Train samples after augmentation: {X_train.shape[0]}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val, snr_val), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

LOW_SNR_BAND = (-12, -6)

model = HybridAMCNet(num_classes=len(mods), num_subbands=8, stem_channels=32,
                      input_length=X_train.shape[-1]).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"HybridAMCNet+ComplexBN trainable parameters: {params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-7)
criterion = nn.CrossEntropyLoss()


def evaluate_val():
    model.eval()
    total_loss, band_loss, band_correct, band_n = 0.0, 0.0, 0, 0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch, snr_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            per_loss = F.cross_entropy(out, y_batch, reduction='none')
            preds = out.argmax(dim=1)
            total_loss += per_loss.sum().item()
            n += y_batch.size(0)
            mask = ((snr_batch >= LOW_SNR_BAND[0]) & (snr_batch <= LOW_SNR_BAND[1])).to(device)
            if mask.any():
                band_loss += per_loss[mask].sum().item()
                band_correct += (preds[mask] == y_batch[mask]).sum().item()
                band_n += mask.sum().item()
    return total_loss / n, band_loss / band_n, 100 * band_correct / band_n


best_band_loss, no_improve = float('inf'), 0
PATIENCE, MAX_EPOCHS = 20, 150

for epoch in range(MAX_EPOCHS):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    agg_loss, band_loss, band_acc = evaluate_val()
    scheduler.step(agg_loss)

    if band_loss < best_band_loss:
        best_band_loss = band_loss
        torch.save(model.state_dict(), 'hybrid_cbn_final.pt')
        no_improve = 0
        marker = " (best)"
    else:
        no_improve += 1
        marker = ""

    print(f"Epoch {epoch+1:3d}: Band({LOW_SNR_BAND[0]}..{LOW_SNR_BAND[1]}dB) "
          f"Loss {band_loss:.4f} (Acc {band_acc:.2f}%){marker}")

    if no_improve >= PATIENCE:
        print(f"\nStopped: no improvement in {PATIENCE} epochs (best was epoch {epoch+1-PATIENCE}).")
        break

model.load_state_dict(torch.load('hybrid_cbn_final.pt', weights_only=True))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
print(f"\n>> Final test accuracy: {100*correct/total:.2f}%  ({params:,} params)")

snr_correct, snr_total = {}, {}
with torch.no_grad():
    for i in range(len(X_test)):
        x = X_test[i].unsqueeze(0).to(device)
        y = y_test[i].item()
        snr = int(snr_test[i])
        pred = model(x).argmax(dim=1).item()
        snr_total[snr] = snr_total.get(snr, 0) + 1
        if pred == y:
            snr_correct[snr] = snr_correct.get(snr, 0) + 1

print("\nAccuracy per SNR:")
for snr in sorted(snr_total.keys()):
    acc = 100 * snr_correct.get(snr, 0) / snr_total[snr]
    print(f"SNR {snr:4d} dB: {acc:.2f}%")