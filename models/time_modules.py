"""
DisTime time encoder and decoder modules.
Based on DisTime paper and InternVL implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TimeDecoder(nn.Module):
    """
    Time Decoder: Converts hidden states at TIME_STAMP positions to time distributions.
    
    Architecture: 3-layer MLP with ReLU activations
    Input: Hidden states at TIME_STAMP token positions
    Output: Distribution logits over time bins + predicted timestamps
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        reg_max: int = 32,
        num_layers: int = 3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reg_max = reg_max
        self.num_layers = num_layers
        
        # Output dimension: 2 * (reg_max + 1) for start and end distributions
        self.output_dim = 2 * (reg_max + 1)
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: project to output dimension
                self.layers.append(nn.Linear(hidden_size, self.output_dim))
            else:
                # Hidden layers
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Project layer for converting distribution to timestamps
        self.project = Project(reg_max=reg_max)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        time_token_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            time_token_mask: (batch, seq_len) boolean mask for TIME_STAMP positions
        
        Returns:
            logits: (N, 2*(reg_max+1)) distribution logits
            pred_times: (N, 2) predicted [start, end] timestamps
        """
        # Extract hidden states at TIME_STAMP positions
        # time_token_mask: (batch, seq_len) -> indices where mask is True
        x = hidden_states[time_token_mask]  # (N, hidden_size)
        
        if x.shape[0] == 0:
            # No TIME_STAMP tokens found
            device = hidden_states.device
            return (
                torch.zeros(0, self.output_dim, device=device),
                torch.zeros(0, 2, device=device)
            )
        x = x.to(self.layers[0].weight.dtype)
        # Forward through MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        logits = x  # (N, 2*(reg_max+1))
        
        # Convert logits to timestamps
        pred_times = self.project(logits)  # (N, 2)
        
        return logits, pred_times


class TimeEncoder(nn.Module):
    """
    Time Encoder: Converts timestamps to embeddings via Gaussian distributions.
    
    Process:
    1. Convert timestamps to Gaussian distributions
    2. Concatenate start and end distributions
    3. Pass through MLP to get embeddings
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        reg_max: int = 32,
        num_layers: int = 3,
        sigma: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reg_max = reg_max
        self.num_layers = num_layers
        self.sigma = sigma
        
        # Input dimension: 2 * (reg_max + 1) for concatenated distributions
        self.input_dim = 2 * (reg_max + 1)
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: from distribution to hidden
                self.layers.append(nn.Linear(self.input_dim, hidden_size))
            else:
                # Hidden layers
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Register anchor points buffer
        self.register_buffer(
            'anchors',
            torch.arange(0, reg_max + 1, dtype=torch.float32)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def generate_gaussian(self, times: torch.Tensor) -> torch.Tensor:
        """
        Generate Gaussian distributions centered at given times.
        
        Args:
            times: (N,) tensor of time values in [0, reg_max]
        
        Returns:
            distributions: (N, reg_max+1) Gaussian distributions
        """
        # times: (N,) -> (N, 1)
        # anchors: (reg_max+1,) -> (1, reg_max+1)
        times = times.unsqueeze(-1)  # (N, 1)
        anchors = self.anchors.to(times.device).unsqueeze(0)  # (1, reg_max+1)
        
        # Compute Gaussian: exp(-0.5 * ((x - mu) / sigma)^2)
        diff = anchors - times  # (N, reg_max+1)
        gaussian = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        
        # Normalize to sum to 1
        gaussian = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)
        
        return gaussian
    
    def forward(
        self,
        start_times: torch.Tensor,
        end_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            start_times: (N,) start timestamps in [0, reg_max]
            end_times: (N,) end timestamps in [0, reg_max]
        
        Returns:
            embeddings: (N, hidden_size) time embeddings
        """
        # Generate Gaussian distributions
        start_dist = self.generate_gaussian(start_times)  # (N, reg_max+1)
        end_dist = self.generate_gaussian(end_times)  # (N, reg_max+1)
        
        # Concatenate distributions
        x = torch.cat([start_dist, end_dist], dim=-1)  # (N, 2*(reg_max+1))
        x = x.to(self.layers[0].weight.dtype)
        # Forward through MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        return x  # (N, hidden_size)


class Project(nn.Module):
    """
    Project distribution logits to timestamps using softmax weighted sum.
    """
    
    def __init__(self, reg_max: int = 32):
        super().__init__()
        self.reg_max = reg_max
        
        # Register anchor points
        self.register_buffer(
            'anchors',
            torch.arange(0, reg_max + 1, dtype=torch.float32)
        )
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, 2*(reg_max+1)) distribution logits
        
        Returns:
            times: (N, 2) predicted [start, end] timestamps
        """
        reg_max_p1 = self.reg_max + 1
        
        # Split into start and end
        start_logits = logits[:, :reg_max_p1]  # (N, reg_max+1)
        end_logits = logits[:, reg_max_p1:]  # (N, reg_max+1)
        
        # Softmax to get probabilities
        start_probs = F.softmax(start_logits, dim=-1)  # (N, reg_max+1)
        end_probs = F.softmax(end_logits, dim=-1)  # (N, reg_max+1)
        
        # Weighted sum with anchors
        anchors = self.anchors.to(start_probs.device)
        start_times = (start_probs * anchors).sum(dim=-1)  # (N,)
        end_times = (end_probs * anchors).sum(dim=-1)  # (N,)
        
        return torch.stack([start_times, end_times], dim=-1)  # (N, 2)


# Test functions
if __name__ == "__main__":
    print("Testing DisTime modules...")
    
    batch_size = 2
    seq_len = 100
    hidden_size = 2048
    reg_max = 32
    
    # Create modules
    decoder = TimeDecoder(hidden_size=hidden_size, reg_max=reg_max)
    encoder = TimeEncoder(hidden_size=hidden_size, reg_max=reg_max)
    
    # Test decoder
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    time_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    time_mask[0, 10] = True  # One TIME_STAMP in first sample
    time_mask[1, 20] = True  # One TIME_STAMP in second sample
    
    logits, pred_times = decoder(hidden_states, time_mask)
    print(f"Decoder output - logits: {logits.shape}, pred_times: {pred_times.shape}")
    print(f"  Predicted times: {pred_times}")
    
    # Test encoder
    start_times = torch.tensor([5.0, 10.0])
    end_times = torch.tensor([15.0, 25.0])
    
    embeddings = encoder(start_times, end_times)
    print(f"Encoder output - embeddings: {embeddings.shape}")
    
    print("All tests passed!")
