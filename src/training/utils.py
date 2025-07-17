# src/training/utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss function.
    Based on FaceNet (Schroff et al., 2015)
    L(A, P, N) = max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Calculate Euclidean distance between anchor and positive
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        # Calculate Euclidean distance between anchor and negative
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss calculation
        loss = torch.relu(dist_pos.pow(2) - dist_neg.pow(2) + self.margin)
        return torch.mean(loss)

def save_checkpoint(state: dict, filepath: str):
    """Saves model and optimizer state to a file."""
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer = None):
    """Loads model and optionally optimizer state from a file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

def set_device():
    """Sets the device to GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device