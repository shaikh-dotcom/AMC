import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch

DATA_PATH = "RML2016.10a_dict.pkl"  # update if your filename differs

# Load raw data
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

mods = sorted(set(k[0] for k in data.keys()))
snrs = sorted(set(k[1] for k in data.keys()))
print("Modulations:", mods)
print("SNRs:", snrs)

# Flatten dict into arrays
X = []       # signals
y = []       # modulation labels (as integers)
snr_list = []  # SNR value for each sample, kept for later per-SNR evaluation

mod_to_idx = {mod: i for i, mod in enumerate(mods)}

for (mod, snr), signals in data.items():
    for sig in signals:
        X.append(sig)
        y.append(mod_to_idx[mod])
        snr_list.append(snr)

X = np.array(X)          # shape: (N, 2, 128)
y = np.array(y)          # shape: (N,)
snr_list = np.array(snr_list)

print(f"\nTotal samples: {X.shape[0]}")
print(f"Signal shape per sample: {X.shape[1:]}")
print(f"Number of classes: {len(mods)}")

# Split into train/test (80/20), stratified by label so class balance is preserved
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, y, snr_list, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print(f"\nX_train tensor shape: {X_train_t.shape}")
print(f"y_train tensor shape: {y_train_t.shape}")

# Save as .pt files so you don't have to redo this every time
torch.save({
    'X_train': X_train_t, 'y_train': y_train_t,
    'X_test': X_test_t, 'y_test': y_test_t,
    'snr_train': snr_train, 'snr_test': snr_test,
    'mods': mods
}, 'amc_data_split.pt')

print("\nSaved split to amc_data_split.pt")