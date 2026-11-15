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
import torchvision.models as models

from .config import ACTConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor of shape (B, T, d_model)."""
        return x + self.pe[:, : x.size(1)]


def _build_2d_sinusoidal_pos_embed(
    d_model: int, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return (1, 1, d_model, height, width) 2D sinusoidal positional embeddings."""
    if d_model % 4 != 0:
        raise ValueError(
            "2D spatial positional encoding requires d_model divisible by 4"
        )
    half = d_model // 2
    y_pos = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)
    x_pos = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, half, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / half)
    )
    pe_y = torch.zeros(height, half, device=device, dtype=torch.float32)
    pe_y[:, 0::2] = torch.sin(y_pos * div)
    pe_y[:, 1::2] = torch.cos(y_pos * div)
    pe_x = torch.zeros(width, half, device=device, dtype=torch.float32)
    pe_x[:, 0::2] = torch.sin(x_pos * div)
    pe_x[:, 1::2] = torch.cos(x_pos * div)
    grid_y = pe_y.unsqueeze(1).expand(height, width, half)
    grid_x = pe_x.unsqueeze(0).expand(height, width, half)
    grid = torch.cat([grid_y, grid_x], dim=-1).permute(2, 0, 1)  # (d_model, H, W)
    return grid.unsqueeze(0).unsqueeze(0).to(dtype=dtype)


class MultiCameraVisualBackbone(nn.Module):
    """Visual feature extraction backbone preserving 2D spatial patch tokens."""

    def __init__(
        self, d_model: int = 256, num_cameras: int = 4, spatial_tokens: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.spatial_tokens = spatial_tokens
        base_resnet = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_proj = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(
        self, images: torch.Tensor, camera_identity: torch.Tensor
    ) -> torch.Tensor:
        """Extract camera tokens of shape (B, num_cameras * patches, d_model)."""
        B, num_cams, C, H, W = images.shape
        feat_map = self.stem(images.view(B * num_cams, C, H, W))  # (B*N, 512, Hf, Wf)
        if not self.spatial_tokens:
            feat_map = self.avgpool(feat_map)  # (B*N, 512, 1, 1)
        projected = self.spatial_proj(feat_map)  # (B*N, d_model, Hf, Wf)
        _, _, Hf, Wf = projected.shape
        tokens = projected.view(B, num_cams, self.d_model, Hf, Wf)
        if self.spatial_tokens:
            pos_2d = _build_2d_sinusoidal_pos_embed(
                self.d_model, Hf, Wf, tokens.device, tokens.dtype
            )
            tokens = tokens + pos_2d
        tokens = tokens + camera_identity.unsqueeze(-1).unsqueeze(-1)
        # (B, num_cams, Hf, Wf, d_model) -> (B, num_cams * Hf * Wf, d_model)
        return tokens.permute(0, 1, 3, 4, 2).reshape(
            B, num_cams * Hf * Wf, self.d_model
        )


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
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_log_var = nn.Linear(d_model, latent_dim)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor, action_mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        seq = torch.cat(
            [cls_tokens, state_tokens, action_tokens], dim=1
        )  # (B, K+2, d_model)
        seq = self.pos_enc(seq)
        padding = None
        if action_mask is not None:
            padding = torch.cat(
                [
                    torch.zeros((B, 2), dtype=torch.bool, device=state.device),
                    ~action_mask.bool(),
                ],
                dim=1,
            )
        encoded = self.transformer_encoder(seq, src_key_padding_mask=padding)

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
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dec_layers
        )
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
    """Custom visual ACT-like policy; upstream LeRobot compatibility unverified."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        config.validate()
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
            spatial_tokens=config.spatial_tokens,
        )

        self.camera_identity = nn.Parameter(
            torch.randn(1, self.num_cameras, config.d_model)
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
        log_var = torch.clamp(log_var, -20.0, 20.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        state: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
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
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("State dimension differs from selected profile")
        if (
            images is None
            or images.ndim != 5
            or images.shape[:3] != (B, self.num_cameras, 3)
        ):
            raise ValueError(
                "Visual ACT requires all four image tensors; no state-only fallback"
            )
        if actions is not None and actions.shape != (
            B,
            self.chunk_size,
            self.action_dim,
        ):
            raise ValueError("Action chunk shape differs from configuration")

        # 1. CVAE Latent variable sampling
        if actions is not None:
            mu, log_var = self.cvae_encoder(state, actions, action_mask)
            z = self.reparameterize(mu, log_var)
        else:
            mu, log_var = None, None
            # During evaluation, set z = 0 (mean of prior)
            z = torch.zeros(
                (B, self.latent_dim), device=state.device, dtype=state.dtype
            )

        # 2. Extract visual features
        camera_tokens = self.visual_backbone(images, self.camera_identity)

        # 3. Assemble memory conditioning sequence
        z_token = self.latent_proj(z).unsqueeze(1)  # (B, 1, d_model)
        state_token = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)

        memory = torch.cat(
            [z_token, state_token, camera_tokens], dim=1
        )  # (B, 2+num_cams, d_model)

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
            l1_loss = (l1_diff * mask).sum() / (
                mask.sum() * target_actions.shape[-1] + 1e-8
            )
        else:
            l1_loss = l1_diff.mean()

        # KL divergence loss
        if mu is not None and log_var is not None:
            log_var = torch.clamp(log_var, -20.0, 20.0)
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
