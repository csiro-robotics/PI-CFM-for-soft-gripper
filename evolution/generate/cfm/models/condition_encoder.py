"""
Spatial Style Encoder v2 (CFM-sim self-contained copy).

Spatially-varying design-space conditioning with soft DS weight maps.

Architecture:
  BC stream:  CNN(channels 0-5) → (B, N, D) spatial tokens
  DS stream:  avg_pool(channels 7-10) to patch grid
              → einsum blend with per-class style embeddings
              → spatial refinement conv
              → (B, N, D) spatial tokens
  Fusion:     bc + ds → LayerNorm → MLP → (B, N, D) context

Returns: context_tokens (B, N, D) — fused BC+DS spatial tokens for cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialStyleEncoder(nn.Module):
    """
    Condition encoder v2: spatially-varying design-space style maps.

    DS channels → soft spatial weight maps → per-patch blended embeddings.
    Supports smooth spatial transitions and CutMix augmentation.
    """

    def __init__(
        self,
        hidden_dim=384,
        img_height=128,
        img_width=64,
        patch_size=4,
        num_ds=4,
        bc_channels=6,
        ds_channel_offset=6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.num_ds = num_ds
        self.bc_channels = bc_channels
        self.ds_channel_offset = ds_channel_offset

        # ---- BC Spatial Encoder ----
        self.bc_cnn = nn.Sequential(
            nn.Conv2d(bc_channels, 64, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.bc_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )

        # ---- DS Spatial Encoder ----
        self.ds_class_embeddings = nn.Parameter(
            torch.zeros(num_ds, hidden_dim)
        )
        self.ds_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.ds_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )

        # ---- Fusion: BC + DS → context tokens ----
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Null Tokens for CFG ----
        self.null_context = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.bc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.ds_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.ds_class_embeddings, std=0.02)
        nn.init.trunc_normal_(self.null_context, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_bc(self, bc):
        """Encode BC spatial maps to patch-aligned tokens."""
        feat = self.bc_cnn(bc)
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens + self.bc_pos_embed

    def encode_ds_spatial(self, ds_maps):
        """Encode spatially-varying DS weight maps to patch-aligned tokens."""
        ds_weights = F.adaptive_avg_pool2d(
            ds_maps, (self.grid_h, self.grid_w)
        )
        ds_weights = ds_weights / (ds_weights.sum(dim=1, keepdim=True) + 1e-8)
        blended = torch.einsum(
            'bkhw,kd->bdhw', ds_weights, self.ds_class_embeddings
        )
        refined = blended + self.ds_refine(blended)
        tokens = refined.flatten(2).transpose(1, 2)
        return tokens + self.ds_pos_embed

    def forward(self, cond_spatial, use_null=False, null_mask=None, disable_ds=False):
        """
        Encode the full condition tensor into context tokens.

        Args:
            cond_spatial: (B, C, H, W) condition tensor.
                channels 0-5:   BC spatial maps.
                channels 6-9:   DS weight maps (broadcast across H×W).
            use_null:    bool — return null tokens (CFG uncond pass).
            null_mask:   (B,) bool — per-sample null mask (CFG dropout).
            disable_ds:  bool — zero out DS stream (BC-only ablation).

        Returns:
            context_tokens: (B, N, D) — fused BC+DS for cross-attention.
        """
        B = cond_spatial.shape[0]

        if use_null:
            return self.null_context.expand(B, -1, -1)

        bc = cond_spatial[:, :self.bc_channels]
        ds_maps = cond_spatial[
            :,
            self.ds_channel_offset:self.ds_channel_offset + self.num_ds,
        ]

        bc_tokens = self.encode_bc(bc)

        if disable_ds:
            ds_tokens = torch.zeros_like(bc_tokens)
        else:
            ds_tokens = self.encode_ds_spatial(ds_maps)

        fused = bc_tokens + ds_tokens
        context_tokens = self.fusion_proj(self.fusion_norm(fused))

        if null_mask is not None and null_mask.any():
            null_ctx = self.null_context.expand(B, -1, -1)
            context_tokens = torch.where(
                null_mask[:, None, None], null_ctx, context_tokens,
            )

        return context_tokens
