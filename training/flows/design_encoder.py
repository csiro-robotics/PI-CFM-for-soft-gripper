"""
Spatial Style Encoder (v2) — spatially-varying design-space conditioning.

v1 (CompositionalConditionEncoder) takes integer DS labels → token bank lookup.
v2 takes soft DS weight maps (B, K, H, W) where K = num_ds classes:
  - Each pixel specifies a blend of design spaces (e.g., 30% finray + 70% topopt)
  - Trained with CutMix augmentation to expose model to spatially mixed layouts
  - At inference, supports smooth spatial style interpolation

Architecture:
  BC stream:  CNN(channels 0-5) → (B, N, D) spatial tokens       [same as v1]
  DS stream:  avg_pool(channels 7-10) to patch grid
              → einsum blend with per-class style embeddings
              → spatial refinement conv
              → (B, N, D) spatial tokens                          [NEW]
  Fusion:     bc + ds → LayerNorm → MLP → (B, N, D) context

Returns: context_tokens (B, N, D) — fused BC+DS spatial tokens for cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialStyleEncoder(nn.Module):
    """
    Condition encoder v2: spatially-varying design-space style maps.

    Drop-in replacement for CompositionalConditionEncoder.

    Key difference from v1:
      v1: DS channels → argmax → integer → token bank lookup → 4 global DS tokens
      v2: DS channels → soft spatial weight maps → per-patch blended embeddings

    This enables:
      - CutMix training: rectangular regions from different DS classes
      - Inference interpolation: "30% finray here, 70% topopt there"
      - Smooth spatial transitions between design styles
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
        self.grid_h = img_height // patch_size   # 32
        self.grid_w = img_width // patch_size    # 16
        self.num_patches = self.grid_h * self.grid_w  # 512
        self.num_ds = num_ds
        self.bc_channels = bc_channels
        self.ds_channel_offset = ds_channel_offset

        # ---- BC Spatial Encoder (same architecture as v1) ----
        # (B, 6, 128, 64) → (B, D, 32, 16) via stride 1→2→2 = 4× downsample
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

        # ---- DS Spatial Encoder (NEW) ----
        # Per-class style embeddings: the "style palette"
        # Each design space gets one D-dim vector.
        # For one-hot maps: equivalent to embedding lookup.
        # For soft maps: smooth linear interpolation between styles.
        self.ds_class_embeddings = nn.Parameter(
            torch.zeros(num_ds, hidden_dim)
        )

        # Spatial refinement after blending.
        # Depthwise-separable conv: captures local spatial structure in the
        # blended DS feature map (e.g., transitions between DS regions).
        # Residual: refined = blended + refine(blended)
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
        """Encode BC spatial maps to patch-aligned tokens.

        Args:
            bc: (B, 6, H, W) boundary condition maps.

        Returns:
            (B, N, D) spatial BC tokens with positional embeddings.
        """
        feat = self.bc_cnn(bc)                          # (B, D, Hp, Wp)
        tokens = feat.flatten(2).transpose(1, 2)        # (B, N, D)
        return tokens + self.bc_pos_embed

    def encode_ds_spatial(self, ds_maps):
        """Encode spatially-varying DS weight maps to patch-aligned tokens.

        Each patch gets a D-dim vector that is a weighted sum of class embeddings,
        refined by a spatial convolution. For uniform one-hot maps (standard training),
        all patches get the same class embedding (equivalent to v1 lookup).

        Args:
            ds_maps: (B, K, H, W) per-pixel DS weights.
                     Values in [0, 1], should sum to 1 per pixel.
                     K = num_ds (typically 4: topopt, finray, graph, lattice).

        Returns:
            (B, N, D) spatial DS tokens with positional embeddings.
        """
        # Downsample weight maps to patch grid
        ds_weights = F.adaptive_avg_pool2d(
            ds_maps, (self.grid_h, self.grid_w)
        )  # (B, K, Hp, Wp)

        # Normalize to sum-to-1 per spatial location (safety)
        ds_weights = ds_weights / (ds_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Spatially blend class embeddings
        # ds_weights: (B, K, Hp, Wp),  ds_class_embeddings: (K, D)
        # → blended: (B, D, Hp, Wp)
        blended = torch.einsum(
            'bkhw,kd->bdhw', ds_weights, self.ds_class_embeddings
        )

        # Spatial refinement (residual)
        refined = blended + self.ds_refine(blended)

        # Flatten to token sequence
        tokens = refined.flatten(2).transpose(1, 2)    # (B, N, D)
        return tokens + self.ds_pos_embed

    def forward(self, cond_spatial, use_null=False, null_mask=None, disable_ds=False):
        """
        Encode the full condition tensor into context tokens.

        Args:
            cond_spatial: (B, C, H, W) condition tensor.
                channels 0-5:   BC spatial maps (support, force, output).
                channels 6-9:   DS weight maps (can be spatially varying).
            use_null:    bool — return null tokens for all samples (CFG uncond pass).
            null_mask:   (B,) bool — per-sample null mask (CFG dropout during training).
            disable_ds:  bool — zero out DS stream (Stage 1 BC-only training).

        Returns:
            context_tokens:  (B, N, D) — fused BC+DS for cross-attention.
        """
        B = cond_spatial.shape[0]

        if use_null:
            return self.null_context.expand(B, -1, -1)

        # Parse channels
        bc = cond_spatial[:, :self.bc_channels]
        ds_maps = cond_spatial[
            :,
            self.ds_channel_offset:self.ds_channel_offset + self.num_ds,
        ]

        # Encode streams
        bc_tokens = self.encode_bc(bc)                        # (B, N, D)

        if disable_ds:
            ds_tokens = torch.zeros_like(bc_tokens)
        else:
            ds_tokens = self.encode_ds_spatial(ds_maps)       # (B, N, D)

        # Fuse: elementwise add → normalize → project
        fused = bc_tokens + ds_tokens
        context_tokens = self.fusion_proj(self.fusion_norm(fused))

        # CFG null masking (per-sample)
        if null_mask is not None and null_mask.any():
            null_ctx = self.null_context.expand(B, -1, -1)
            context_tokens = torch.where(
                null_mask[:, None, None], null_ctx, context_tokens,
            )

        return context_tokens


class ConditionDecoder(nn.Module):
    """Small decoder to reconstruct condition tensor from spatial tokens.

    Used ONLY for standalone encoder pretraining validation.
    Reconstructs: spatial_tokens → feature map → deconv → (B, out_channels, H, W)
    """

    def __init__(self, out_channels=10, hidden_dim=384, img_height=128, img_width=64):
        super().__init__()
        self.grid_h = img_height // 4
        self.grid_w = img_width // 4

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, spatial_tokens):
        """spatial_tokens: (B, N, D) → (B, out_channels, H, W)"""
        B, N, D = spatial_tokens.shape
        feat = spatial_tokens.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        return self.decoder(feat)
