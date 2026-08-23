import torch
from torch.utils.data import DataLoader, TensorDataset
from hybrid_model_cbn import HybridAMCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ---------------------------------------------------------
# Load shifted test sets
# ---------------------------------------------------------
data = torch.load("shifted_test_sets.pt", weights_only=False)

conditions = data["conditions"]
y_test = data["y_test"]
mods = data["mods"]

print("Available test conditions:")
for name, X in conditions.items():
    print(f"  {name:12s}: {tuple(X.shape)}")

# ---------------------------------------------------------
# Build the SAME model architecture used during training
# ---------------------------------------------------------
model = HybridAMCNet(
    num_classes=len(mods),
    num_subbands=8,
    stem_channels=32,
    input_length=128
).to(device)

# Load trained weights
model.load_state_dict(
    torch.load("hybrid_cbn_final.pt", weights_only=True)
)

model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {params:,}")
print("Loaded: hybrid_cbn_final.pt\n")


# ---------------------------------------------------------
# Evaluate one condition
# ---------------------------------------------------------
def evaluate_condition(X, y):
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=256,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            preds = output.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return 100.0 * correct / total


# ---------------------------------------------------------
# Overall domain-shift accuracy
# ---------------------------------------------------------
print("=" * 60)
print("DOMAIN-SHIFT TEST")
print("=" * 60)

results = {}

for name, X in conditions.items():
    acc = evaluate_condition(X, y_test)
    results[name] = acc

    print(f"{name:15s}: {acc:.2f}%")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for name, acc in results.items():
    print(f"{name:15s} {acc:.2f}%")