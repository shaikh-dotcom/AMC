import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from hybrid_model import HybridAMCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

data = torch.load('amc_data_split.pt', weights_only=False)
X_train_full, y_train_full = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
mods = data['mods']
snr_test = data['snr_test']

# Same split as before -- comparable to the earlier ablation run.
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

val_loader = DataLoader(TensorDataset(X_val, y_val, snr_val), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

# The band this whole experiment is about: exclude near-chance bins
# (-20/-18/-16 dB) that only add noise to checkpoint selection.
BAND_MIN_DB = -12
BAND_MAX_DB = -6


def rotate_augment(X, y):
    I, Q = X[:, 0, :], X[:, 1, :]
    variants = [
        torch.stack([I, Q], dim=1),
        torch.stack([Q, -I], dim=1),
        torch.stack([-I, -Q], dim=1),
        torch.stack([-Q, I], dim=1),
    ]
    return torch.cat(variants, dim=0), y.repeat(4)


def evaluate_val(loader):
    """Returns (agg_loss, agg_acc, band_loss, band_acc) on the val set,
    where 'band' = only samples with BAND_MIN_DB <= SNR <= BAND_MAX_DB."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    band_loss, band_correct, band_n = 0.0, 0, 0
    criterion_sum = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for X_batch, y_batch, snr_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            per_sample_loss = criterion_sum(outputs, y_batch)
            preds = outputs.argmax(dim=1)

            total_loss += per_sample_loss.sum().item()
            total_correct += (preds == y_batch).sum().item()
            total_n += y_batch.size(0)

            band_mask = ((snr_batch >= BAND_MIN_DB) & (snr_batch <= BAND_MAX_DB)).to(device)
            if band_mask.any():
                band_loss += per_sample_loss[band_mask].sum().item()
                band_correct += (preds[band_mask] == y_batch[band_mask]).sum().item()
                band_n += band_mask.sum().item()

    agg_loss = total_loss / total_n
    agg_acc = 100 * total_correct / total_n
    b_loss = band_loss / band_n if band_n else float('nan')
    b_acc = 100 * band_correct / band_n if band_n else float('nan')
    return agg_loss, agg_acc, b_loss, b_acc


def evaluate_test(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return 100 * correct / total


def per_snr_accuracy():
    model.eval()
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
    return {snr: 100 * snr_correct.get(snr, 0) / snr_total[snr] for snr in sorted(snr_total)}


def run_config(name, use_aug, use_bn, patience=20, max_epochs=150):
    global model
    print(f"\n{'='*60}\nConfig: {name}  (augmentation={use_aug}, batchnorm={use_bn})\n"
          f"Checkpoint selection band: {BAND_MIN_DB} to {BAND_MAX_DB} dB\n{'='*60}")

    torch.manual_seed(42)

    if use_aug:
        X_train, y_train = rotate_augment(X_train_base, y_train_base)
    else:
        X_train, y_train = X_train_base, y_train_base

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    model = HybridAMCNet(num_classes=len(mods), num_subbands=8,
                          input_length=X_train.shape[-1], use_batchnorm=use_bn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    best_band_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_path = f'ablation_narrowband_{name}.pt'

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        agg_loss, agg_acc, band_loss, band_acc = evaluate_val(val_loader)
        scheduler.step(agg_loss)  # LR schedule still follows overall stability

        if band_loss < best_band_loss:
            best_band_loss = band_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
            marker = " (best-band)"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(f"  Epoch {epoch+1:3d}: Val Acc {agg_acc:.2f}%, "
              f"Band({BAND_MIN_DB}..{BAND_MAX_DB}dB) Loss {band_loss:.4f} (Acc {band_acc:.2f}%){marker}")

        if epochs_without_improvement >= patience:
            print(f"  Stopped: no band improvement in {patience} epochs (best was epoch {epoch+1-patience}).")
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_acc = evaluate_test(test_loader)
    snr_acc = per_snr_accuracy()

    print(f"  >> Final test accuracy: {test_acc:.2f}%")
    return {'name': name, 'test_acc': test_acc, 'snr_acc': snr_acc}


model = None
configs = [
    ("plain",         False, False),
    ("aug_only",      True,  False),
    ("bn_only",        False, True),
    ("aug_and_bn",     True,  True),
]

results = []
for name, use_aug, use_bn in configs:
    results.append(run_config(name, use_aug, use_bn))

print(f"\n{'='*60}\nSUMMARY (narrow-band checkpoint selection: {BAND_MIN_DB} to {BAND_MAX_DB} dB)\n{'='*60}")
print(f"{'Config':<15} {'Test Acc':>10}")
for r in results:
    print(f"{r['name']:<15} {r['test_acc']:>9.2f}%")

print(f"\n{'Config':<15}" + "".join(f"{snr:>7}" for snr in sorted(results[0]['snr_acc'].keys())))
for r in results:
    row = "".join(f"{r['snr_acc'][snr]:>6.1f}%" for snr in sorted(r['snr_acc'].keys()))
    print(f"{r['name']:<15}{row}")