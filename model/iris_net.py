import torch
import torch.nn as nn

class IrisNet(nn.Module):
    def __init__(self, num_classes):
        super(IrisNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (16, 50, 50)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (32, 25, 25)
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.cnn(x)
