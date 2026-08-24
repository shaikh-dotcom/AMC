import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


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


class HybridAMCNet(nn.Module):
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
        x = self.front_bn(self.front_end(x))
        x = self.stem(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        return self.fc(self.gdwconv(x))


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
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val, snr_val), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

LOW_SNR_BAND = (-12, -6)


def evaluate_val(model):
    model.eval()
    band_loss, band_correct, band_n = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch, snr_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            per_loss = F.cross_entropy(out, y_batch, reduction='none')
            preds = out.argmax(dim=1)
            mask = ((snr_batch >= LOW_SNR_BAND[0]) & (snr_batch <= LOW_SNR_BAND[1])).to(device)
            if mask.any():
                band_loss += per_loss[mask].sum().item()
                band_correct += (preds[mask] == y_batch[mask]).sum().item()
                band_n += mask.sum().item()
    return (band_loss / band_n, 100 * band_correct / band_n) if band_n else (float('nan'), float('nan'))


def evaluate_test(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return 100 * correct / total


def per_snr_accuracy(model):
    model.eval()
    correct, total = {}, {}
    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i].unsqueeze(0).to(device)
            y = y_test[i].item()
            snr = int(snr_test[i])
            pred = model(x).argmax(dim=1).item()
            total[snr] = total.get(snr, 0) + 1
            if pred == y:
                correct[snr] = correct.get(snr, 0) + 1
    return {s: 100 * correct.get(s, 0) / total[s] for s in sorted(total)}


def run_config(name, num_subbands, stem_channels, patience=20, max_epochs=150):
    print(f"\n{'='*60}\n{name}: num_subbands={num_subbands}, stem_channels={stem_channels}\n{'='*60}")
    torch.manual_seed(42)
    model = HybridAMCNet(num_classes=len(mods), num_subbands=num_subbands,
                          stem_channels=stem_channels, input_length=X_train.shape[-1]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()
    best_band_loss, no_improve = float('inf'), 0
    ckpt = f'lever_final_{name}.pt'

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        agg_loss = 0.0
        model.eval()
        with torch.no_grad():
            n = 0
            for X_batch, y_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                agg_loss += F.cross_entropy(model(X_batch), y_batch, reduction='sum').item()
                n += y_batch.size(0)
            agg_loss /= n
        scheduler.step(agg_loss)

        band_loss, band_acc = evaluate_val(model)
        if band_loss < best_band_loss:
            best_band_loss = band_loss
            torch.save(model.state_dict(), ckpt)
            no_improve = 0
            marker = " (best)"
        else:
            no_improve += 1
            marker = ""

        print(f"  Epoch {epoch+1:3d}: Band Acc {band_acc:.2f}%{marker}")

        if no_improve >= patience:
            print(f"  Stopped at epoch {epoch+1} (best was epoch {epoch+1-patience}).")
            break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_acc = evaluate_test(model)
    snr_acc = per_snr_accuracy(model)
    print(f"  >> Test Acc: {test_acc:.2f}%  |  params: {params:,}")
    return {'name': name, 'params': params, 'test_acc': test_acc, 'snr_acc': snr_acc}


configs = [
    ("backend_cut_full",  8, 32),
    ("frontend_cut_full", 4, 64),
]

results = [run_config(*c) for c in configs]

print(f"\n{'='*60}\nFULL-RIGOR SUMMARY (patience=20, matches 'original' protocol)\n{'='*60}")
print(f"{'Config':<20}{'Params':>9}{'TestAcc':>10}")
for r in results:
    print(f"{r['name']:<20}{r['params']:>9,}{r['test_acc']:>9.2f}%")

snrs = sorted(results[0]['snr_acc'].keys())
print(f"\n{'Config':<20}" + "".join(f"{s:>6}" for s in snrs))
for r in results:
    print(f"{r['name']:<20}" + "".join(f"{r['snr_acc'][s]:>5.1f}%" for s in snrs))