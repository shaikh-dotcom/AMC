import torch
from torch.utils.data import DataLoader, TensorDataset
from hybrid_model import HybridAMCNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}\n")


# =========================================================
# CHANGE THESE TWO VALUES
# =========================================================

CHECKPOINT = "ablation_narrowband_bn_only.pt"
USE_BATCHNORM = True

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
    print(
        f"  {name:12s}: {tuple(X.shape)}"
    )


# =========================================================
# Build EXACT SAME architecture
# =========================================================

model = HybridAMCNet(
    num_classes=len(mods),
    num_subbands=8,
    input_length=128,
    use_batchnorm=USE_BATCHNORM
).to(device)


# =========================================================
# Parameter count
# =========================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nHybridAMCNet parameters: "
    f"{num_params:,}"
)


# =========================================================
# Load checkpoint
# =========================================================

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        weights_only=True
    )
)

model.eval()

print(
    f"Loaded: {CHECKPOINT}"
)

print(
    f"BatchNorm: {USE_BATCHNORM}\n"
)


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

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    return 100.0 * correct / total


# =========================================================
# DOMAIN-SHIFT TEST
# =========================================================

print("=" * 60)
print("HYBRIDAM CNET ABLATION DOMAIN-SHIFT TEST")
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
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SUMMARY")
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


# =========================================================
# Parameter information
# =========================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(
    f"Checkpoint       : {CHECKPOINT}"
)

print(
    f"BatchNorm        : {USE_BATCHNORM}"
)

print(
    f"Subbands         : 8"
)

print(
    f"Parameters       : {num_params:,}"
)