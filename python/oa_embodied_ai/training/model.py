# SPDX-License-Identifier: MIT
"""Vision-based Action Chunking with Transformers (ACT) model architecture.

Deliverable C11: ACT Imitation Learning Baseline.
Resolves C10->C11 handoff findings:
2. Vision-Based Architecture: Implement true multi-view visual ACT policy
   using the 4 canonical camera streams alongside the proprioceptive state.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import ACTConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor of shape (B, T, d_model)."""
        return x + self.pe[:, : x.size(1)]


class MultiCameraVisualBackbone(nn.Module):
    """Visual feature extraction backbone for canonical 4-camera streams."""

    def __init__(self, d_model: int = 256, num_cameras: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # Shared ResNet-18 visual encoder
        base_resnet = models.resnet18(weights=None)
        # Remove fc layer
        base_resnet.fc = nn.Identity()
        self.backbone = base_resnet

        # Linear projection from ResNet-18 (512-dim) to transformer d_model
        self.proj = nn.Linear(512, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera feature tokens.

        Args:
            images: Tensor of shape (B, num_cameras, 3, H, W).

        Returns:
            camera_tokens: Tensor of shape (B, num_cameras, d_model).
        """
        B, num_cams, C, H, W = images.shape
        flat_imgs = images.view(B * num_cams, C, H, W)
        features = self.backbone(flat_imgs)  # (B * num_cams, 512)
        tokens = self.proj(features)  # (B * num_cams, d_model)
        tokens = tokens.view(B, num_cams, self.d_model)
        return tokens


class CVAEEncoder(nn.Module):
    """CVAE Encoder mapping (state, action_chunk) -> latent distribution (mu, log_var)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        latent_dim: int = 32,
        n_heads: int = 8,
        enc_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_log_var = nn.Linear(d_model, latent_dim)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state and target actions into latent distribution parameters.

        Args:
            state: (B, state_dim)
            actions: (B, K, action_dim)

        Returns:
            (mu, log_var): both of shape (B, latent_dim)
        """
        B = state.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        state_tokens = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
        action_tokens = self.action_proj(actions)  # (B, K, d_model)

        seq = torch.cat([cls_tokens, state_tokens, action_tokens], dim=1)  # (B, K+2, d_model)
        seq = self.pos_enc(seq)
        encoded = self.transformer_encoder(seq)

        cls_out = encoded[:, 0]  # (B, d_model)
        mu = self.to_mu(cls_out)
        log_var = self.to_log_var(cls_out)
        return mu, log_var


class TransformerDecoder(nn.Module):
    """Transformer Decoder outputting multi-step action chunks (K x action_dim)."""

    def __init__(
        self,
        action_dim: int,
        chunk_size: int = 100,
        d_model: int = 256,
        n_heads: int = 8,
        dec_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model

        # Fixed learned action queries for K steps
        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Decode action sequence conditioned on memory tokens.

        Args:
            memory: Condition tokens (B, seq_len, d_model).

        Returns:
            actions: Predicted action sequence (B, chunk_size, action_dim).
        """
        B = memory.shape[0]
        queries = self.query_embed.expand(B, -1, -1)  # (B, K, d_model)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        pred_actions = self.action_head(decoded)  # (B, K, action_dim)
        return pred_actions


class ACTPolicy(nn.Module):
    """Complete Vision-Based Action Chunking with Transformers (ACT) Policy."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.latent_dim = config.latent_dim
        self.d_model = config.d_model
        self.num_cameras = len(config.camera_keys)

        # Multi-camera visual backbone
        self.visual_backbone = MultiCameraVisualBackbone(
            d_model=config.d_model,
            num_cameras=self.num_cameras,
        )

        # Proprioceptive state & latent projections for memory
        self.state_proj = nn.Linear(config.state_dim, config.d_model)
        self.latent_proj = nn.Linear(config.latent_dim, config.d_model)

        # CVAE Encoder
        self.cvae_encoder = CVAEEncoder(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            latent_dim=config.latent_dim,
            n_heads=config.n_heads,
            enc_layers=config.enc_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dec_layers=config.dec_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample latent z using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        state: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass of ACT policy.

        Args:
            state: Proprioceptive joint/base states (B, state_dim).
            images: Multi-camera images (B, num_cameras, 3, H, W).
            actions: Target action chunks (B, K, action_dim) for training mode.

        Returns:
            pred_actions: (B, chunk_size, action_dim)
            mu: (B, latent_dim) or None
            log_var: (B, latent_dim) or None
        """
        B = state.shape[0]

        # 1. CVAE Latent variable sampling
        if actions is not None:
            mu, log_var = self.cvae_encoder(state, actions)
            z = self.reparameterize(mu, log_var)
        else:
            mu, log_var = None, None
            # During evaluation, set z = 0 (mean of prior)
            z = torch.zeros((B, self.latent_dim), device=state.device, dtype=state.dtype)

        # 2. Extract visual features
        if images is not None:
            camera_tokens = self.visual_backbone(images)  # (B, num_cameras, d_model)
        else:
            # Fallback zero tokens if images omitted in state-only tests
            camera_tokens = torch.zeros(
                (B, self.num_cameras, self.d_model),
                device=state.device,
                dtype=state.dtype,
            )

        # 3. Assemble memory conditioning sequence
        z_token = self.latent_proj(z).unsqueeze(1)  # (B, 1, d_model)
        state_token = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)

        memory = torch.cat([z_token, state_token, camera_tokens], dim=1)  # (B, 2+num_cams, d_model)

        # 4. Decode action chunk
        pred_actions = self.decoder(memory)
        return pred_actions, mu, log_var

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mu: Optional[torch.Tensor],
        log_var: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        l1_weight: float = 1.0,
        kl_weight: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined imitation L1 and CVAE KL divergence loss."""
        # L1 loss
        l1_diff = torch.abs(pred_actions - target_actions)
        if action_mask is not None:
            # Mask shape (B, K) -> expand to (B, K, action_dim)
            mask = action_mask.unsqueeze(-1)
            l1_loss = (l1_diff * mask).sum() / (mask.sum() * target_actions.shape[-1] + 1e-8)
        else:
            l1_loss = l1_diff.mean()

        # KL divergence loss
        if mu is not None and log_var is not None:
            # D_KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            kl_div = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
            kl_loss = kl_div.mean()
        else:
            kl_loss = torch.tensor(0.0, device=pred_actions.device)

        total_loss = l1_loss * l1_weight + kl_loss * kl_weight
        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "kl_loss": kl_loss,
        }

