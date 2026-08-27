"""
Diffusion Transformer (DiT) for Flow Matching.

Based on Lumina-T2I (Flag-DiT / Alpha-VLLM, 2024), adapted for PICFM
topology optimization with SpatialStyleEncoder (v2).

Architecture (Lumina-T2I):
  - RMSNorm (instead of LayerNorm)
  - SwiGLU FFN (3-matrix gated, instead of GELU MLP)
  - RoPE positional encoding (1D with EOL row separators)
  - QK-Norm: LayerNorm on Q, K, and cross-K
  - Fused self-attention + tanh-gated cross-attention (zero-init)
  - Per-block adaLN MLP (zero-init)
  - adaln_input = t_emb + cap_emb
  - attention_y_norm: RMSNorm on context tokens before cross-attn K/V
  - EOL tokens for row separation in 1D RoPE sequence

Conditioning (Lumina pattern — no separate global_vec):
  cond_spatial (B, 10, H, W)
    → SpatialStyleEncoder → context_tokens (B, N_ctx, D)
      → cross-attention K/V          (per-patch spatial conditioning)
      → mean-pool → cap_embedder → adaLN  (global conditioning)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.condition_encoder import SpatialStyleEncoder


# ======================================================================
# Components
# ======================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no centering, learnable scale)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift. shift/scale are (B, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ======================================================================
# Embedding Layers
# ======================================================================

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings (from GLIDE)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ======================================================================
# Core DiT Blocks (Lumina architecture)
# ======================================================================

class Attention(nn.Module):
    """
    Fused self-attention + tanh-gated cross-attention (Lumina-T2I).

    Self-attention: Q,K,V from image tokens, with RoPE and QK-Norm.
    Cross-attention: same Q, separate K,V from context tokens, with tanh gate.
    Both outputs go through single wo projection.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        y_dim: int = 0,
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Self-attention projections (no bias -- Lumina convention)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        # Cross-attention projections (separate K,V from context)
        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros([self.n_heads]))

        # Shared output projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # QK-Norm (LayerNorm on Q, K, and cross-K)
        if qk_norm:
            self.q_norm = nn.LayerNorm(n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to self-attention Q and K (NOT cross-attention K)."""
        with torch.amp.autocast("cuda", enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        # QK-Norm on self-attention Q,K
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE on self-attention Q,K only
        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        # GQA repeat if n_kv_heads < n_heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Self-attention via scaled_dot_product_attention
        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(
                -1, self.n_heads, seqlen, -1
            ),
        ).permute(0, 2, 1, 3).to(dtype)

        # Cross-attention (separate K,V from context tokens)
        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).view(
                bsz, -1, self.n_kv_heads, self.head_dim
            )
            yv = self.wv_y(y).view(bsz, -1, self.n_kv_heads, self.head_dim)
            if n_rep > 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                attn_mask=y_mask.view(bsz, 1, 1, -1).expand(
                    bsz, self.n_heads, seqlen, -1
                ),
            ).permute(0, 2, 1, 3)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_y

        output = output.flatten(-2)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN (Lumina-T2I): w2(SiLU(w1(x)) * w3(x))."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Lumina-T2I transformer block:
      1. Per-block adaLN MLP → shift/scale/gate for SA and FFN (zero-init)
      2. RMSNorm → modulate → Fused self-attn + tanh-gated cross-attn
      3. RMSNorm → modulate → SwiGLU FFN
      4. attention_y_norm: RMSNorm on context before cross-attn K/V
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        y_dim: int,
        ffn_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=ffn_hidden_dim if ffn_hidden_dim is not None else 4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        # Per-block adaLN MLP (Lumina: zero-init -> gates start at 0)
        adaln_dim = min(dim, 1024)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # RMSNorm on context tokens before cross-attention K/V
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                x_mask,
                freqs_cis,
                self.attention_y_norm(y),
                y_mask,
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp),
            )
        else:
            x = x + self.attention(
                self.attention_norm(x),
                x_mask,
                freqs_cis,
                self.attention_y_norm(y),
                y_mask,
            )
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    """Final layer: per-layer adaLN MLP (zero-init) -> LayerNorm -> Linear (zero-init)."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        adaln_dim = min(hidden_size, 1024)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_dim, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ======================================================================
# DiT Model
# ======================================================================

class DiTModel(nn.Module):
    """
    Diffusion Transformer with Lumina-T2I architecture for conditional flow matching.

    Conditioning follows the Lumina pattern exactly:
      cond_spatial → SpatialStyleEncoder → context_tokens (B, N_ctx, D)
        → cross-attention K/V   (per-patch spatial conditioning)
        → mean-pool → cap_embedder → adaln_input  (global conditioning)

    No separate global_vec — the adaLN signal is derived from the same
    context tokens via mean-pooling, identical to how Lumina pools cap_feats.
    """

    def __init__(
        self,
        img_height=128,
        img_width=64,
        patch_size=4,
        in_channels=1,
        out_channels=1,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        cond_spatial_channels=10,
        num_ds=4,
        bc_channels=6,
        ds_channel_offset=6,
        n_kv_heads=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        qk_norm=True,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.cond_spatial_channels = cond_spatial_channels

        dim = hidden_dim
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # ---- Image patch embedding (MAE-style: reshape + Linear) ----
        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels, dim, bias=True
        )

        # ---- EOL/PAD tokens (Lumina row-end markers for 1D RoPE) ----
        self.eol_token = nn.Parameter(torch.empty(dim))
        self.pad_token = nn.Parameter(torch.empty(dim))

        # ---- Timestep embedding (sinusoidal -> MLP) ----
        adaln_dim = min(dim, 1024)
        self.t_embedder = TimestepEmbedder(adaln_dim)

        # ---- Condition embedding (Lumina: LN → Linear, zero-init) ----
        # Input: mean-pooled context tokens (dim)
        # Output: adaln_dim for adaLN modulation
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, adaln_dim, bias=True),
        )

        # ---- Condition encoder (SpatialStyleEncoder v2) ----
        self.cond_encoder = SpatialStyleEncoder(
            hidden_dim=hidden_dim,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
            num_ds=num_ds,
            bc_channels=bc_channels,
            ds_channel_offset=ds_channel_offset,
        )

        # ---- Transformer blocks (Lumina architecture) ----
        ffn_hidden = int(mlp_ratio * dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                layer_id=i,
                dim=dim,
                n_heads=num_heads,
                n_kv_heads=n_kv_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                y_dim=dim,
                ffn_hidden_dim=ffn_hidden,
            )
            for i in range(depth)
        ])

        # ---- Final layer ----
        self.final_layer = FinalLayer(dim, patch_size, out_channels)

        # ---- RoPE frequencies ----
        self.freqs_cis = DiTModel.precompute_freqs_cis(dim // num_heads, 4096)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.eol_token, std=0.02)
        nn.init.normal_(self.pad_token, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.cap_embedder[1].weight)
        nn.init.zeros_(self.cap_embedder[1].bias)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[1].weight)
            nn.init.zeros_(layer.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def patchify_and_embed(self, x):
        pH = pW = self.patch_size
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.x_embedder(x)
        x = torch.cat(
            [x, self.eol_token.view(1, 1, 1, -1).expand(B, H // pH, 1, -1)],
            dim=2,
        )
        x = x.flatten(1, 2)
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)
        return x, mask, [(H, W)] * B

    def unpatchify(self, x, img_size):
        pH = pW = self.patch_size
        H, W = img_size[0]
        B = x.size(0)
        x = x[:, : (H // pH) * (W // pW + 1)]
        x = x.view(B, H // pH, W // pW + 1, pH, pW, self.out_channels)
        x = x[:, :, :-1]
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return x

    def forward(self, t, x, cond_spatial=None, use_null=False, null_mask=None, disable_ds=False):
        """
        Forward pass: predict velocity field v_θ(t, x_t, conditions).

        Args:
            t:            (B,)             timestep in [0, 1]
            x:            (B, 1, H, W)    noisy image x_t
            cond_spatial: (B, C, H, W)    full condition tensor (11 channels)
            use_null:     bool            use null tokens (CFG unconditional pass)
            null_mask:    (B,) bool       per-sample null mask (CFG dropout)
            disable_ds:   bool            skip DS tokens (Stage 1)

        Returns:
            v: (B, 1, H, W) predicted velocity
        """
        while t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        # ---- Encode conditions → context tokens ----
        if cond_spatial is not None:
            if cond_spatial.dim() == 3:
                cond_spatial = cond_spatial.unsqueeze(0)
            context = self.cond_encoder(
                cond_spatial, use_null=use_null, null_mask=null_mask,
                disable_ds=disable_ds,
            )
        else:
            context = self.cond_encoder(
                torch.zeros(x.shape[0], self.cond_spatial_channels,
                            self.img_height, self.img_width,
                            device=x.device, dtype=x.dtype),
                use_null=True,
            )

        # ---- Lumina pattern: pool context → cap_embedder → adaLN ----
        cap_feats_pool = context.mean(dim=1)              # (B, D)
        cap_emb = self.cap_embedder(cap_feats_pool)       # (B, adaln_dim)

        # ---- Context mask (all valid — no padding) ----
        context_mask = torch.ones(
            context.shape[0], context.shape[1],
            dtype=torch.bool, device=context.device,
        )

        # ---- Patchify image ----
        x, x_mask, img_size = self.patchify_and_embed(x)
        self.freqs_cis = self.freqs_cis.to(x.device)

        # ---- adaLN input: timestep + pooled context ----
        t_emb = self.t_embedder(t)
        adaln_input = t_emb + cap_emb

        # ---- Transformer ----
        for layer in self.layers:
            x = layer(x, x_mask, context, context_mask,
                      self.freqs_cis[:x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        return self.unpatchify(x, img_size)



# ======================================================================
# Model constructors
# ======================================================================

def DiT_S_4(**kwargs):
    """DiT-Small with patch size 4: D=384, 12 layers, 6 heads."""
    return DiTModel(hidden_dim=384, depth=12, num_heads=6, patch_size=4, **kwargs)

def DiT_B_4(**kwargs):
    """DiT-Base with patch size 4: D=768, 12 layers, 12 heads."""
    return DiTModel(hidden_dim=768, depth=12, num_heads=12, patch_size=4, **kwargs)

def DiT_S_8(**kwargs):
    """DiT-Small with patch size 8: D=384, 12 layers, 6 heads."""
    return DiTModel(hidden_dim=384, depth=12, num_heads=6, patch_size=8, **kwargs)
