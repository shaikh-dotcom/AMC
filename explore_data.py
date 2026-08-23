import pickle
import numpy as np
import matplotlib.pyplot as plt

# Update this path to match your actual file location/name
DATA_PATH = "RML2016.10a_dict.pkl"

# Load the dataset
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1') # encoding needed for Python 2 pickle files

# The data is a dict: keys are (modulation_type, SNR) tuples
# Let's look at what keys exist
keys = list(data.keys())
print(f"Total number of (modulation, SNR) combinations: {len(keys)}")
print(f"First 5 keys: {keys[5]}")

# Pick one example: e.g., first key in the list
example_key = keys[1]
mod_type, snr = example_key
signals = data[example_key]  # shape: (num_samples, 2, 128)

print(f"\nExample: modulation={mod_type}, SNR={snr}")
print(f"Shape of signals array: {signals.shape}")

# Plot one signal (I and Q channels)
one_signal = signals[20]  # shape: (2, 128) -> row 0 = I, row 1 = Q
I = one_signal[0]
Q = one_signal[1]

plt.figure(figsize=(10, 4))
plt.plot(I, label='I (in-phase)')
plt.plot(Q, label='Q (quadrature)')
plt.title(f"Signal sample: {mod_type} at {snr} dB SNR")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.legend()
plt.show()