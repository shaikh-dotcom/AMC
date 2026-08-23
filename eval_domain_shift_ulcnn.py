import torch
from torch.utils.data import DataLoader, TensorDataset
from ulcnn_model import ULCNN


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# =========================================================
# Load domain-shift test sets
# =========================================================
data = torch.load(
    "shifted_test_sets.pt",
    weights_only=False
)

conditions = data["conditions"]
y_test = data["y_test"]
mods = data["mods"]


print("Available test conditions:")
for name, X in conditions.items():
    print(f"  {name:12s}: {tuple(X.shape)}")


# =========================================================
# Build EXACT SAME ULCNN architecture used for training
# =========================================================
model = ULCNN(
    num_classes=len(mods),
    n_neuron=16,
    n_mobileunit=6,
    kernel_size=5
).to(device)


# =========================================================
# Load trained ULCNN checkpoint
# =========================================================
model.load_state_dict(
    torch.load(
        "ulcnn_faithful_best.pt",
        weights_only=True
    )
)

model.eval()


num_params = sum(
    p.numel() for p in model.parameters()
)

print(f"\nULCNN parameters: {num_params:,}")
print("Loaded: ulcnn_faithful_best.pt\n")


# =========================================================
# Evaluate one condition
# =========================================================
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

            outputs = model(X_batch)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# =========================================================
# Domain-shift evaluation
# =========================================================
print("=" * 60)
print("ULCNN DOMAIN-SHIFT TEST")
print("=" * 60)

results = {}

for name, X in conditions.items():

    accuracy = evaluate_condition(
        X,
        y_test
    )

    results[name] = accuracy

    print(
        f"{name:15s}: "
        f"{accuracy:.2f}%"
    )


# =========================================================
# Summary
# =========================================================
print("\n" + "=" * 60)
print("ULCNN DOMAIN-SHIFT SUMMARY")
print("=" * 60)

clean_acc = results["clean"]

for name, accuracy in results.items():

    if name == "clean":
        print(
            f"{name:15s} "
            f"{accuracy:.2f}%"
        )

    else:
        drop = clean_acc - accuracy

        print(
            f"{name:15s} "
            f"{accuracy:.2f}% "
            f"(drop {drop:.2f} pp)"
        )