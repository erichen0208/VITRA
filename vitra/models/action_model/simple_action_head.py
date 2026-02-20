"""
simple_action_head.py

Lightweight MLP action head for small robot action spaces.
Replaces the full DiT diffusion head when the action space is small
(e.g. 6-DOF joint + gripper) and diffusion is overkill.

Input:  cognition token z [B, D_llm]  +  optional state [B, state_dim]
Output: action chunk [B, chunk_size, action_dim]
Loss:   MSE on predicted vs ground-truth action chunks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleActionHead(nn.Module):
    """MLP action head: VLM cognition token → action chunk."""

    def __init__(
        self,
        token_size: int = 2304,
        action_dim: int = 6,
        chunk_size: int = 16,
        state_dim: int = 0,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim

        in_dim = token_size + state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, chunk_size * action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """
        z:     [B, token_size]  — cognition token from VLM
        state: [B, state_dim]   — optional proprioceptive state
        returns: [B, chunk_size, action_dim]
        """
        if state is not None and self.state_dim > 0:
            z = torch.cat([z, state], dim=-1)
        return self.net(z).view(-1, self.chunk_size, self.action_dim)

    def loss(self, z: torch.Tensor, target: torch.Tensor,
             state: torch.Tensor = None) -> dict:
        """
        target: [B, chunk_size, action_dim] — ground-truth actions (normalized)
        Returns dict with "loss" key for compatibility with VITRA training loop.
        """
        pred = self.forward(z, state)
        return {"loss": F.mse_loss(pred, target)}

    @torch.no_grad()
    def predict(self, z: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """Inference: returns [B, chunk_size, action_dim]."""
        return self.forward(z, state)
