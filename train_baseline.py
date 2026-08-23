import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from baseline_model import BaselineCNN

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the split data
data = torch.load('amc_data_split.pt', weights_only=False)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
mods = data['mods']

# Wrap into DataLoaders (handles batching automatically)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize model, loss function, optimizer
model = BaselineCNN(num_classes=len(mods)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
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

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation on test set
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
print(f"\nTest Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'baseline_cnn.pt')
print("Model saved to baseline_cnn.pt")
# Per-SNR accuracy breakdown
model.eval()
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