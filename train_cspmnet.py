import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from CSPMNet import CSPMNet


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}\n")


# =========================================================
# Load dataset
# =========================================================

data = torch.load(
    "amc_data_split.pt",
    weights_only=False
)

X_train_full = data["X_train"]
y_train_full = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

mods = data["mods"]

snr_train_full = np.asarray(
    data["snr_train"]
)

snr_test = data["snr_test"]


# =========================================================
# Same train/validation split as HybridAMCNet and ULCNN
# =========================================================

(
    X_train,
    X_val,
    y_train,
    y_val,
    snr_train,
    snr_val
) = train_test_split(
    X_train_full.numpy(),
    y_train_full.numpy(),
    snr_train_full,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full.numpy()
)


X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

snr_val = torch.tensor(
    snr_val,
    dtype=torch.long
)


# =========================================================
# Same I/Q rotation augmentation
# =========================================================

def rotate_augment(X, y):

    I = X[:, 0, :]
    Q = X[:, 1, :]

    variants = [

        torch.stack(
            [I, Q],
            dim=1
        ),

        torch.stack(
            [Q, -I],
            dim=1
        ),

        torch.stack(
            [-I, -Q],
            dim=1
        ),

        torch.stack(
            [-Q, I],
            dim=1
        ),
    ]

    X_aug = torch.cat(
        variants,
        dim=0
    )

    y_aug = y.repeat(4)

    return X_aug, y_aug


X_train, y_train = rotate_augment(
    X_train,
    y_train
)


print(
    f"Train samples after rotation augmentation: "
    f"{X_train.shape[0]}"
)

print(
    f"Validation samples: {X_val.shape[0]}"
)


# =========================================================
# Data loaders
# =========================================================

train_loader = DataLoader(
    TensorDataset(
        X_train,
        y_train
    ),
    batch_size=256,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        X_val,
        y_val,
        snr_val
    ),
    batch_size=256,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(
        X_test,
        y_test
    ),
    batch_size=256,
    shuffle=False
)


# =========================================================
# CSPMNet
# =========================================================

model = CSPMNet(
    input_dim=[2, 128],
    output_dim=len(mods)
).to(device)


# =========================================================
# Parameter count
# =========================================================

param_info = model.count_parameters()

num_params = param_info["total_trainable"]

print(
    f"\nCSPMNet trainable parameters: "
    f"{num_params:,}"
)

print(
    f"Number of subbands: "
    f"{param_info['num_subbands']}"
)

print(
    f"Lags: "
    f"{param_info['lags']}"
)

print(
    f"Feature channels: "
    f"{param_info['feature_channels']}"
)


# =========================================================
# Training configuration
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=10,
    min_lr=1e-7
)


# Same narrow SNR checkpoint band
LOW_SNR_BAND = (-12, -6)


# Same early stopping as ULCNN
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 35

best_band_loss = float("inf")
epochs_without_improvement = 0


# =========================================================
# Validation
# =========================================================

def evaluate_val(loader):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    band_loss = 0.0
    band_correct = 0
    band_n = 0

    with torch.no_grad():

        for (
            X_batch,
            y_batch,
            snr_batch
        ) in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            _, outputs = model(
                X_batch
            )

            per_sample_loss = F.cross_entropy(
                outputs,
                y_batch,
                reduction="none"
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_loss += (
                per_sample_loss.sum().item()
            )

            total_correct += (
                predictions == y_batch
            ).sum().item()

            total_n += y_batch.size(0)


            # Narrow-band validation
            band_mask = (
                (snr_batch >= LOW_SNR_BAND[0])
                &
                (snr_batch <= LOW_SNR_BAND[1])
            ).to(device)

            if band_mask.any():

                band_loss += (
                    per_sample_loss[band_mask]
                    .sum()
                    .item()
                )

                band_correct += (
                    predictions[band_mask]
                    == y_batch[band_mask]
                ).sum().item()

                band_n += (
                    band_mask.sum().item()
                )


    agg_loss = total_loss / total_n

    agg_acc = (
        100.0
        * total_correct
        / total_n
    )

    b_loss = (
        band_loss / band_n
        if band_n
        else float("nan")
    )

    b_acc = (
        100.0
        * band_correct
        / band_n
        if band_n
        else float("nan")
    )

    return (
        agg_loss,
        agg_acc,
        b_loss,
        b_acc
    )


# =========================================================
# Training
# =========================================================

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        _, outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_loss = (
        running_loss
        / len(train_loader)
    )


    # Validation
    (
        val_loss,
        val_acc,
        band_loss,
        band_acc
    ) = evaluate_val(val_loader)


    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # Best checkpoint based on narrow SNR band
    if band_loss < best_band_loss:

        best_band_loss = band_loss

        torch.save(
            model.state_dict(),
            "cspmnet_faithful_best.pt"
        )

        marker = " (best-band)"

        epochs_without_improvement = 0

    else:

        marker = ""

        epochs_without_improvement += 1


    print(
        f"Epoch {epoch+1:3d}: "
        f"Val Acc {val_acc:.2f}%, "
        f"Band({LOW_SNR_BAND[0]}.."
        f"{LOW_SNR_BAND[1]}dB) "
        f"Loss {band_loss:.4f} "
        f"(Acc {band_acc:.2f}%)"
        f"{marker}"
    )


    if (
        epochs_without_improvement
        >= EARLY_STOP_PATIENCE
    ):

        print(
            f"\nNo band improvement in "
            f"{EARLY_STOP_PATIENCE} epochs -- "
            f"stopping early at epoch "
            f"{epoch+1}."
        )

        break


# =========================================================
# Load best model
# =========================================================

model.load_state_dict(
    torch.load(
        "cspmnet_faithful_best.pt",
        weights_only=True
    )
)

model.eval()


# =========================================================
# Overall test accuracy
# =========================================================

correct = 0
total = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        _, outputs = model(
            X_batch
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


test_accuracy = (
    100.0
    * correct
    / total
)


print(
    f"\n>> Final CSPMNet test accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f">> Trainable parameters: "
    f"{num_params:,}"
)


# =========================================================
# Accuracy per SNR
# =========================================================

snr_correct = {}
snr_total = {}


with torch.no_grad():

    for i in range(
        len(X_test)
    ):

        x = (
            X_test[i]
            .unsqueeze(0)
            .to(device)
        )

        y = y_test[i].item()

        snr = int(
            snr_test[i]
        )

        _, output = model(
            x
        )

        pred = torch.argmax(
            output,
            dim=1
        ).item()


        snr_total[snr] = (
            snr_total.get(snr, 0)
            + 1
        )

        if pred == y:

            snr_correct[snr] = (
                snr_correct.get(snr, 0)
                + 1
            )


print(
    "\nAccuracy per SNR:"
)

for snr in sorted(
    snr_total.keys()
):

    acc = (
        100.0
        * snr_correct.get(snr, 0)
        / snr_total[snr]
    )

    print(
        f"SNR {snr:4d} dB: "
        f"{acc:.2f}%"
    )