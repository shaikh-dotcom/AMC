import pickle
import numpy as np
import torch

# --- Compare raw signal amplitude scale between the two datasets ---
with open('RML2016.10a_dict.pkl', 'rb') as f:
    data_a = pickle.load(f, encoding='latin1')
with open('RML2016.10b.dat', 'rb') as f:  # match your actual filename
    data_b = pickle.load(f, encoding='latin1')

# Compare the SAME modulation at the SAME SNR across both datasets
for snr in [-20, -10, 0, 18]:
    key_a = ('QPSK', snr)
    key_b = ('QPSK', snr)
    sig_a = np.array(data_a[key_a])
    sig_b = np.array(data_b[key_b])
    print(f"SNR {snr:4d} dB | 10a mean|amp|: {np.abs(sig_a).mean():.6f}  "
          f"10b mean|amp|: {np.abs(sig_b).mean():.6f}  "
          f"ratio: {np.abs(sig_b).mean() / np.abs(sig_a).mean():.3f}")

# --- Check prediction bias at -20dB: is the model predicting one class repeatedly? ---
from hybrid_model_cbn import HybridAMCNet
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = torch.load('amc_data_split.pt', weights_only=False)
mods_train = split['mods']

model = HybridAMCNet(num_classes=len(mods_train), num_subbands=8,
                      stem_channels=32, input_length=128).to(device)
model.load_state_dict(torch.load('hybrid_cbn_final.pt', weights_only=True))
model.eval()

sig_20 = torch.tensor(np.array(data_b[('QPSK', -20)]), dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(sig_20).argmax(dim=1).cpu().numpy()

pred_counts = Counter(preds)
print(f"\nAt -20dB, true label = QPSK. Model's predicted class distribution:")
for idx, count in pred_counts.most_common():
    print(f"  {mods_train[idx]:8s}: {count} times ({100*count/len(preds):.1f}%)")