import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from hybrid_model import HybridAMCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = torch.load('amc_data_split.pt', weights_only=False)
X_train_full, y_train_full = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
mods = data['mods']

# NOTE: this assumes an 'snr_train' key parallel to 'snr_test', giving the
# SNR (dB) of every training sample. If your saved .pt file uses a
# different key (or doesn't have one), replace this line accordingly.
snr_train_full = np.asarray(data['snr_train'])

# Carve a validation set out of the training set. The val set is only used
# to decide the LR schedule and which checkpoint to keep — it never gets
# rotated/augmented, so it stays representative of the real distribution.
# snr_train_full is split alongside X/y (same indices) so we can identify
# which validation samples are low-SNR later.
X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
    X_train_full.numpy(), y_train_full.numpy(), snr_train_full,
    test_size=0.15, random_state=42, stratify=y_train_full.numpy()
)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
snr_val = torch.tensor(snr_val, dtype=torch.long)


def rotate_augment(X, y):
    """90/180/270-degree I/Q phase rotation, stacked with the original.
    Quadruples the training set — the same trick the ULCNN paper uses,
    justified because a phase rotation doesn't change the modulation label."""
    I, Q = X[:, 0, :], X[:, 1, :]
    variants = [
        torch.stack([I, Q], dim=1),
        torch.stack([Q, -I], dim=1),
        torch.stack([-I, -Q], dim=1),
        torch.stack([-Q, I], dim=1),
    ]
    X_aug = torch.cat(variants, dim=0)
    y_aug = y.repeat(4)
    return X_aug, y_aug


X_train, y_train = rotate_augment(X_train, y_train)
print(f"Train samples after rotation augmentation: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val, snr_val), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

model = HybridAMCNet(num_classes=len(mods), num_subbands=8, input_length=X_train.shape[-1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-7)

num_params = sum(p.numel() for p in model.parameters())
print(f"HybridAMCNet trainable parameters: {num_params:,}")

# Bins at or below this SNR are the ones the paper's low-SNR-robustness
# claim actually depends on. Aggregate val loss is dominated by the easier
# high-SNR bins (see the -20..-6 dB dip vs 0..18 dB plateau in your test
# results), so "best checkpoint by aggregate loss" was quietly selecting
# for high-SNR performance. Tracking this separately fixes that.
LOW_SNR_MAX_DB = -6


def evaluate_val(loader):
    """Returns (agg_loss, agg_acc, low_snr_loss, low_snr_acc) on the val set."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    low_loss, low_correct, low_n = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch, snr_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            per_sample_loss = F.cross_entropy(outputs, y_batch, reduction='none')
            preds = outputs.argmax(dim=1)

            total_loss += per_sample_loss.sum().item()
            total_correct += (preds == y_batch).sum().item()
            total_n += y_batch.size(0)

            low_mask = (snr_batch <= LOW_SNR_MAX_DB).to(device)
            if low_mask.any():
                low_loss += per_sample_loss[low_mask].sum().item()
                low_correct += (preds[low_mask] == y_batch[low_mask]).sum().item()
                low_n += low_mask.sum().item()

    agg_loss = total_loss / total_n
    agg_acc = 100 * total_correct / total_n
    low_snr_loss = low_loss / low_n if low_n else float('nan')
    low_snr_acc = 100 * low_correct / low_n if low_n else float('nan')
    return agg_loss, agg_acc, low_snr_loss, low_snr_acc


# 4x the data per epoch means each epoch takes roughly 4x as long as before —
# expect this to run considerably longer than the 10-epoch version. Fine to
# leave running in the background.
num_epochs = 100
best_low_snr_loss = float('inf')

# Stop once low-SNR val loss hasn't improved in this many epochs, instead
# of always burning the full 100. Based on the last run, meaningful
# improvement stalled well before epoch 100 — this just stops paying for
# epochs that aren't helping the metric you actually select on.
EARLY_STOP_PATIENCE = 35
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    val_loss, val_acc, low_snr_loss, low_snr_acc = evaluate_val(val_loader)

    # LR schedule still follows aggregate val loss (overall training
    # stability) — only checkpoint *selection* changes below.
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if low_snr_loss < best_low_snr_loss:
        best_low_snr_loss = low_snr_loss
        torch.save(model.state_dict(), 'hybrid_amcnet_best.pt')
        marker = " (best-lowSNR)"
        epochs_without_improvement = 0
    else:
        marker = ""
        epochs_without_improvement += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f} (Acc {val_acc:.2f}%), "
          f"LowSNR(<={LOW_SNR_MAX_DB}dB) Loss: {low_snr_loss:.4f} (Acc {low_snr_acc:.2f}%), "
          f"LR: {current_lr:.2e}{marker}")

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\nNo low-SNR improvement in {EARLY_STOP_PATIENCE} epochs — "
              f"stopping early at epoch {epoch+1}.")
        break

# Reload the best checkpoint by validation loss before final testing —
# not whatever the last epoch happened to land on.
model.load_state_dict(torch.load('hybrid_amcnet_best.pt', weights_only=True))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy (best checkpoint): {accuracy:.2f}%")

# Per-SNR accuracy breakdown
snr_test = data['snr_test']
snr_correct = {}
snr_total = {}
with torch.no_grad():
    for i in range(len(X_test)):
        x = X_test[i].unsqueeze(0).to(device)
        y = y_test[i].item()
        snr = int(snr_test[i])
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        snr_total[snr] = snr_total.get(snr, 0) + 1
        if pred == y:
            snr_correct[snr] = snr_correct.get(snr, 0) + 1

print("\nAccuracy per SNR:")
for snr in sorted(snr_total.keys()):
    acc = 100 * snr_correct.get(snr, 0) / snr_total[snr]
    print(f"SNR {snr:4d} dB: {acc:.2f}%")