import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(BaselineCNN, self).__init__()
        # Input shape: (batch, 2, 128) -> 2 channels (I and Q), 128 time steps
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 128 -> 64

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 64 -> 32

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # 32 -> 16

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x