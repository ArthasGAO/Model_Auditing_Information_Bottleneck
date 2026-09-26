import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_MLP(nn.Module):
    """
    Simple MLP model for MNIST classification (PyTorch version).
    Input:  1×28×28 grayscale images
    Output: 10-class logits
    """

    def __init__(self, in_channels=1, num_classes=10):
        super(MNIST_MLP, self).__init__()

        # Input layer: flatten 28×28 → 784
        self.fc1 = nn.Linear(28 * 28 * in_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation
        self.relu = nn.ReLU(inplace=True)
        # Optional: can add dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        # Optional: softmax/log-softmax handled outside (e.g. in loss)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: tensor of shape (batch_size, 1, 28, 28)
        Returns:
            logits of shape (batch_size, num_classes)
        """
        # Flatten: (B, 1, 28, 28) → (B, 784)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
