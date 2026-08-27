"""
CFM Design Generator

A self-contained generator class that loads a trained PICFM checkpoint,
reads boundary conditions from a source npz file, generates design
samples, and optionally saves them as npz / mesh files into a
``validation/gen_XXXXX/`` directory structure.

Usage:
    from generator import DesignGenerator

    gen = DesignGenerator(
        checkpoint_path="checkpoints/picfm_dit.pt",
        bc_npz_path="assets/bc_data_00000.npz",
    )

    # Generate samples (returns list of result dicts)
    results = gen.generate(n_samples=4, blend=(0.5, 0.5, 0.0, 0.0))

    # Access raw numpy geometry (128, 64) in [0, 1]
    geom = results[0]["geometry"]

    # Save all results into  validation/gen_00000/, gen_00001/, ...
    gen.save(results, output_root="validation")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Ensure this package's own modules (models/, sampling) are importable
_CFM_DIR = Path(__file__).resolve().parent
if str(_CFM_DIR) not in sys.path:
    sys.path.insert(0, str(_CFM_DIR))

from models.dit import DiTModel
from sampling import euler_sample_dit, euler_sample_compositional


# ---------------------------------------------------------------------------
#  Boundary-condition loader (reads a single npz, mirrors dataset.py logic)
# ---------------------------------------------------------------------------

# Channel layout (11 total):
#   0  BCx          Fixed BC mask (x)
#   1  BCy          Fixed BC mask (y)
#   2  inputx       Prescribed displacement (x)
#   3  inputy       Prescribed displacement (y)
#   4  outputx      Output DOF indicator (x)
#   5  outputy      Output DOF indicator (y)
#   6  w_topopt     Design-space blend weight (broadcast across H×W)
#   7  w_finray     Design-space blend weight (broadcast across H×W)
#   8  w_graph      Design-space blend weight (broadcast across H×W)
#   9  w_lattice    Design-space blend weight (broadcast across H×W)

BC_CHANNEL_NAMES = ["BCx", "BCy", "inputx", "inputy", "outputx", "outputy"]
DESIGN_SPACES = ["topopt", "finray", "graph", "lattice"]
N_COND_CHANNELS = 10


def load_condition_from_npz(
    npz_path: Union[str, Path],
    design_space: str = "topopt",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Read a single ``data_XXXXX.npz`` and return tensors ready for the model.

    Returns
    -------
    geometry : (1, H, W)  float tensor — ground-truth topology in [0, 1]
    condition : (10, H, W) float tensor — full condition stack
    meta : dict — scalar metadata from the npz (volume_fraction, etc.)
    """
    data = np.load(str(npz_path), allow_pickle=True)

    # Ground-truth geometry
    geom = torch.tensor(data["geometry"].astype(np.float32), dtype=dtype).unsqueeze(0)
    H, W = geom.shape[1], geom.shape[2]

    # 6 BC channels
    bc = np.stack([data[ch].astype(np.float32) for ch in BC_CHANNEL_NAMES], axis=0)
    cond = torch.tensor(bc, dtype=dtype)  # (6, H, W)

    # Channels 6-9 — design-space one-hot (broadcast across H×W)
    ds = torch.zeros(len(DESIGN_SPACES), H, W, dtype=dtype)
    if design_space in DESIGN_SPACES:
        ds[DESIGN_SPACES.index(design_space), :, :] = 1.0
    else:
        ds[0, :, :] = 1.0  # default topopt

    cond = torch.cat([cond, ds], dim=0)  # (10, H, W)

    vol_frac = float(data.get("volume_fraction", 0.3))
    meta = {
        "volume_fraction": vol_frac,
        "objective_value": float(data.get("objective_value", 0.0)),
        "penal": float(data.get("penal", 0.0)),
        "rmin": float(data.get("rmin", 0.0)),
        "disp_mag": float(data.get("disp_mag", 0.0)),
    }
    return geom, cond, meta


# ---------------------------------------------------------------------------
#  Model loader
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Union[str, Path], device: torch.device):
    """Load a trained DiT model from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = DiTModel(
        img_height=ckpt_args.get("img_height", 128),
        img_width=ckpt_args.get("img_width", 64),
        patch_size=ckpt_args.get("patch_size", 4),
        in_channels=1,
        out_channels=1,
        hidden_dim=ckpt_args.get("hidden_dim", 384),
        depth=ckpt_args.get("depth", 12),
        num_heads=ckpt_args.get("num_heads", 6),
        mlp_ratio=ckpt_args.get("mlp_ratio", 4.0),
        cond_spatial_channels=ckpt_args.get("cond_spatial_channels", 10),
        num_ds=ckpt_args.get("num_ds", 4),
        bc_channels=ckpt_args.get("bc_channels", 6),
        ds_channel_offset=ckpt_args.get("ds_channel_offset", 6),
    ).to(device)

    if ckpt.get("ema_shadow") is not None:
        model.load_state_dict(ckpt["ema_shadow"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
#  TopologyGenerator
# ---------------------------------------------------------------------------

class DesignGenerator:
    """
    High-level wrapper: load model + BC, generate topologies, save results.

    Parameters
    ----------
    checkpoint_path : path to a ``checkpoint_XXXXXXX.pt`` file.
    bc_npz_path : path to a ``data_XXXXX.npz`` that supplies boundary
        conditions (BCx, BCy, inputx/y, outputx/y, volume_fraction …).
    design_space : one of ``"topopt"``, ``"finray"``, ``"graph"``,
        ``"lattice"`` — sets the default one-hot design-space channel.
    device : ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
    seed : random seed for reproducibility.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        bc_npz_path: Union[str, Path],
        design_space: str = "topopt",
        device: Optional[str] = None,
        seed: int = 42,
        cfg_scale: float = 1.0,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.cfg_scale = cfg_scale
        self.checkpoint_path = Path(checkpoint_path)
        self.bc_npz_path = Path(bc_npz_path)
        self.design_space = design_space

        # Load boundary condition
        self.gt_geom, self.base_cond, self.meta = load_condition_from_npz(
            self.bc_npz_path, design_space=design_space,
        )
        # Keep originals on CPU; send working copies to device when needed
        self._bc_data = np.load(str(self.bc_npz_path), allow_pickle=True)

        # Load model
        print(f"[DesignGenerator] Loading checkpoint: {self.checkpoint_path.name}")
        self.model = _load_model(self.checkpoint_path, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[DesignGenerator] DiT  |  "
              f"{n_params:,} params  |  device={self.device}")
        print(f"[DesignGenerator] BC source: {self.bc_npz_path.name}  "
              f"vol_frac={self.meta['volume_fraction']:.3f}")
        if self.cfg_scale != 1.0:
            print(f"[DesignGenerator] CFG scale: {self.cfg_scale}")

    # ----- core generation -------------------------------------------------

    def generate(
        self,
        n_samples: int = 1,
        blend: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        n_ode_steps: int = 100,
        seed: Optional[int] = None,
        threshold: Optional[float] = 0.5,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate topology samples.

        Parameters
        ----------
        n_samples : number of independent samples to draw.
        blend : (topopt, finray, graph, lattice) design-space weights.
            Overrides channels 7-10 of the condition tensor.
        n_ode_steps : Euler integration steps (higher → better quality).
        seed : per-call random seed (defaults to ``self.seed``).
        threshold : if not None, also produce a binarised geometry at this
            level.  Set to ``None`` to skip.

        Returns
        -------
        List of dicts, one per sample, each containing at least::

            {
                "geometry":      np.ndarray (H, W) float in [0, 1],
                "geometry_raw":  np.ndarray (H, W) float (un-clipped model output),
                "geometry_bin":  np.ndarray (H, W) uint8 0/1  (if threshold set),
                "BCx": ..., "BCy": ...,          # from source npz
                "inputx": ..., "inputy": ...,
                "outputx": ..., "outputy": ...,
                "volume_fraction": float,
                "blend": (T, F, G, L),
            }
        """
        if seed is None:
            seed = self.seed

        # Expand base condition to batch
        cond = self.base_cond.unsqueeze(0).expand(n_samples, -1, -1, -1).clone()
        cond = cond.to(self.device)

        # Override design-space blend channels (channels 6-9 after VF removal)
        cond[:, 6, :, :] = blend[0]
        cond[:, 7, :, :] = blend[1]
        cond[:, 8, :, :] = blend[2]
        cond[:, 9, :, :] = blend[3]

        H, W = cond.shape[2], cond.shape[3]

        torch.manual_seed(seed)
        with torch.no_grad():
            samples = euler_sample_dit(
                self.model,
                cond,
                shape=(n_samples, 1, H, W),
                n_steps=n_ode_steps,
                device=str(self.device),
                cfg_scale=self.cfg_scale,
            )  # (n_samples, 1, H, W) clamped to [0, 1]

        # Package each sample into a result dict
        results: List[Dict] = []
        for i in range(n_samples):
            raw = samples[i, 0].cpu().numpy()                    # (H, W)
            geom_01 = np.clip(raw, 0.0, 1.0).astype(np.float64) # clipped to [0,1]

            res: Dict[str, object] = {
                "geometry": geom_01,
                "geometry_raw": raw.astype(np.float64),
                "blend": blend,
            }

            if threshold is not None:
                res["geometry_bin"] = (geom_01 >= threshold).astype(np.uint8)

            # Carry over BC arrays from the source npz
            for ch in BC_CHANNEL_NAMES:
                res[ch] = self._bc_data[ch]

            # Scalar metadata
            res["volume_fraction"] = self.meta["volume_fraction"]
            res["objective_value"] = self.meta["objective_value"]
            res["penal"]           = self.meta["penal"]
            res["rmin"]            = self.meta["rmin"]
            res["disp_mag"]        = self.meta["disp_mag"]

            results.append(res)

        return results

    # ----- compositional generation ----------------------------------------

    def generate_compositional(
        self,
        n_samples: int = 1,
        regions: List[Dict] = None,
        n_ode_steps: int = 100,
        seed: Optional[int] = None,
        threshold: Optional[float] = 0.5,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate topology samples with spatial design-space composition.

        Uses Multi-condition CFG: each spatial region gets its own DS
        condition, and condition deltas are composed spatially. The model
        sees the full shared x_t at every step, so self-attention
        naturally connects structures across region boundaries.

        Parameters
        ----------
        n_samples : number of independent samples to draw.
        regions : list of dicts, each specifying a spatial region::

            [
                {"blend": (1.0, 0.0, 0.0, 0.0), "mask": "top"},
                {"blend": (0.0, 1.0, 0.0, 0.0), "mask": "bottom"},
            ]

            Each dict has:
              - ``blend``: (topopt, finray, graph, lattice) DS weights.
              - ``mask``: either a string shortcut or a numpy array.

            String shortcuts for ``mask``:
              ``"top"``, ``"bottom"``, ``"left"``, ``"right"``
              — binary splits along height or width.

            Array masks: ``np.ndarray`` of shape ``(H, W)`` with values
            in [0, 1].  Masks should sum to ~1.0 at each pixel.

        n_ode_steps : Euler integration steps.
        seed : random seed (defaults to ``self.seed``).
        threshold : binarisation threshold (``None`` to skip).

        Returns
        -------
        Same format as :meth:`generate`, with ``"blend"`` set to
        ``"compositional"`` and an additional ``"regions"`` key.
        """
        if regions is None:
            regions = [
                {"blend": (1.0, 0.0, 0.0, 0.0), "mask": "top"},
                {"blend": (0.0, 1.0, 0.0, 0.0), "mask": "bottom"},
            ]

        if seed is None:
            seed = self.seed

        H, W = self.base_cond.shape[1], self.base_cond.shape[2]

        # Build condition tensors and spatial masks for each region
        conditions = []
        spatial_masks = []

        for region in regions:
            blend = region["blend"]
            mask_spec = region["mask"]

            # Build condition tensor with this region's DS blend
            cond = self.base_cond.unsqueeze(0).expand(n_samples, -1, -1, -1).clone()
            cond = cond.to(self.device)
            cond[:, 6, :, :] = blend[0]
            cond[:, 7, :, :] = blend[1]
            cond[:, 8, :, :] = blend[2]
            cond[:, 9, :, :] = blend[3]
            conditions.append(cond)

            # Parse spatial mask
            if isinstance(mask_spec, str):
                mask = self._make_named_mask(mask_spec, H, W)
            else:
                mask = torch.tensor(mask_spec, dtype=torch.float32)
            # Shape: (1, 1, H, W) for broadcasting over batch and channels
            mask = mask.view(1, 1, H, W).to(self.device)
            spatial_masks.append(mask)

        torch.manual_seed(seed)
        with torch.no_grad():
            samples = euler_sample_compositional(
                self.model,
                conditions,
                spatial_masks,
                shape=(n_samples, 1, H, W),
                n_steps=n_ode_steps,
                device=str(self.device),
                cfg_scale=self.cfg_scale,
            )

        # Package results
        results: List[Dict] = []
        for i in range(n_samples):
            raw = samples[i, 0].cpu().numpy()
            geom_01 = np.clip(raw, 0.0, 1.0).astype(np.float64)

            res: Dict[str, object] = {
                "geometry": geom_01,
                "geometry_raw": raw.astype(np.float64),
                "blend": "compositional",
                "regions": regions,
            }
            if threshold is not None:
                res["geometry_bin"] = (geom_01 >= threshold).astype(np.uint8)

            for ch in BC_CHANNEL_NAMES:
                res[ch] = self._bc_data[ch]

            res["volume_fraction"] = self.meta["volume_fraction"]
            res["objective_value"] = self.meta["objective_value"]
            res["penal"]           = self.meta["penal"]
            res["rmin"]            = self.meta["rmin"]
            res["disp_mag"]        = self.meta["disp_mag"]

            results.append(res)

        return results

    @staticmethod
    def _make_named_mask(name: str, H: int, W: int) -> torch.Tensor:
        """Create a named binary (or soft) spatial mask of shape (H, W).

        Supported names:
          - 'top', 'bottom', 'left', 'right'
          - 'diagonal', 'anti_diagonal'
          - 'radial' (inner disc), 'radial_outer' (complement)
          - 'checkerboard', 'checkerboard_inv'
          - 'horizontal_bands', 'horizontal_bands_inv'
          - 'spiral'  (complement: pass `1.0 - mask` as numpy array)
        """
        mask = torch.zeros(H, W)
        if name == "top":
            mask[H // 2:, :] = 1.0
        elif name == "bottom":
            mask[:H // 2, :] = 1.0
        elif name == "left":
            mask[:, :W // 2] = 1.0
        elif name == "right":
            mask[:, W // 2:] = 1.0
        elif name == "diagonal":
            for r in range(H):
                cutoff = int(W * r / H)
                mask[r, :cutoff] = 1.0
        elif name == "anti_diagonal":
            for r in range(H):
                cutoff = int(W * r / H)
                mask[r, cutoff:] = 1.0
        elif name in ("radial", "radial_outer"):
            yy, xx = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing="ij",
            )
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            r_max = float(min(H, W)) / 2.0  # inscribed disc radius
            inner = (r <= r_max * 0.5).float()
            mask = inner if name == "radial" else (1.0 - inner)
        elif name in ("checkerboard", "checkerboard_inv"):
            block = max(1, min(H, W) // 8)
            yy, xx = torch.meshgrid(
                torch.arange(H), torch.arange(W), indexing="ij",
            )
            cb = (((yy // block) + (xx // block)) % 2).float()
            mask = cb if name == "checkerboard" else (1.0 - cb)
        elif name in ("horizontal_bands", "horizontal_bands_inv"):
            band = max(1, H // 8)
            rows = torch.arange(H)
            bands = ((rows // band) % 2).float().view(H, 1).expand(H, W).contiguous()
            mask = bands if name == "horizontal_bands" else (1.0 - bands)
        elif name == "spiral":
            yy, xx = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing="ij",
            )
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            dy, dx = yy - cy, xx - cx
            r = torch.sqrt(dy * dy + dx * dx)
            theta = torch.atan2(dy, dx)
            r_norm = r / max(float(min(H, W)) / 2.0, 1.0)
            # Two-arm Archimedean spiral
            phase = torch.sin(theta * 2.0 + r_norm * 6.283)
            mask = (phase > 0).float()
        else:
            raise ValueError(
                f"Unknown mask name '{name}'. Supported: top, bottom, left, "
                f"right, diagonal, anti_diagonal, radial, radial_outer, "
                f"checkerboard, checkerboard_inv, horizontal_bands, "
                f"horizontal_bands_inv, spiral. Or pass a numpy array."
            )
        return mask

    # ----- persistence -----------------------------------------------------

    @staticmethod
    def save(
        results: List[Dict],
        output_root: Union[str, Path] = "validation",
        start_index: int = 0,
        save_npz: bool = True,
        save_mesh: bool = False,
        mesh_extrude_depth: int = 4,
        mesh_voxel_size: float = 1.0,
    ) -> List[Path]:
        """
        Save generated results to ``<output_root>/gen_XXXXX/``.

        Each sub-folder receives:
        - ``data.npz``   — same key layout as the training data
        - ``geometry.png`` — quick visual preview
        - ``geometry.stl``  (optional, if ``save_mesh=True``)

        Parameters
        ----------
        results : output of :meth:`generate`.
        output_root : top-level output directory.
        start_index : numbering offset for gen_XXXXX folders.
        save_npz : write ``data.npz`` per sample.
        save_mesh : convert binarised geometry to STL (requires the
            geometry to be extruded into a thin 3-D slab).
        mesh_extrude_depth : number of voxel layers when extruding 2-D
            geometry to 3-D for STL export.
        mesh_voxel_size : physical size of one voxel edge for STL.

        Returns
        -------
        List of created directory paths.
        """
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        dirs: List[Path] = []

        for i, res in enumerate(results):
            idx = start_index + i
            out_dir = root / f"gen_{idx:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            dirs.append(out_dir)

            geom = res["geometry"]  # (H, W) float64 [0, 1]

            # --- npz (same keys as training data) ---
            if save_npz:
                npz_dict = {
                    "geometry": geom,
                    "BCx": res["BCx"],
                    "BCy": res["BCy"],
                    "inputx": res["inputx"],
                    "inputy": res["inputy"],
                    "outputx": res["outputx"],
                    "outputy": res["outputy"],
                    "objective_value": np.float64(res.get("objective_value", 0.0)),
                    "volume_fraction": np.float64(res.get("volume_fraction", 0.3)),
                    "penal": np.float64(res.get("penal", 0.0)),
                    "rmin": np.float64(res.get("rmin", 0.0)),
                    "disp_mag": np.float64(res.get("disp_mag", 0.0)),
                }
                np.savez(str(out_dir / "data.npz"), **npz_dict)

            # --- preview PNG ---
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(geom, cmap="gray_r", origin="lower")
                axes[0].set_title("Generated (continuous)")
                axes[0].axis("off")
                if "geometry_bin" in res:
                    axes[1].imshow(res["geometry_bin"], cmap="gray_r", origin="lower")
                    axes[1].set_title("Binarised")
                else:
                    axes[1].imshow(geom >= 0.5, cmap="gray_r", origin="lower")
                    axes[1].set_title("Binarised (default 0.5)")
                axes[1].axis("off")
                blend = res.get("blend", (0, 0, 0, 0))
                plt.suptitle(
                    f"T={blend[0]:.2f}  F={blend[1]:.2f}  "
                    f"G={blend[2]:.2f}  L={blend[3]:.2f}",
                    fontsize=11,
                )
                plt.tight_layout()
                plt.savefig(str(out_dir / "geometry.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except ImportError:
                pass

            # --- STL mesh (optional) ---
            if save_mesh:
                geom_bin = res.get(
                    "geometry_bin", (geom >= 0.5).astype(np.uint8)
                )
                _save_stl(geom_bin, out_dir / "geometry.stl",
                          extrude_depth=mesh_extrude_depth,
                          voxel_size=mesh_voxel_size)

            print(f"  [save] {out_dir}")

        return dirs

    # ----- convenience helpers ---------------------------------------------

    def generate_and_save(
        self,
        n_samples: int = 1,
        blend: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        n_ode_steps: int = 100,
        output_root: Union[str, Path] = "validation",
        start_index: int = 0,
        **save_kw,
    ) -> Tuple[List[Dict], List[Path]]:
        """Generate *and* save in one call.  Returns (results, dirs)."""
        results = self.generate(
            n_samples=n_samples, blend=blend, n_ode_steps=n_ode_steps,
        )
        dirs = self.save(
            results, output_root=output_root,
            start_index=start_index, **save_kw,
        )
        return results, dirs


# ---------------------------------------------------------------------------
#  STL export helper (extrudes 2-D binary image into thin 3-D voxel slab)
# ---------------------------------------------------------------------------

def _save_stl(
    binary_2d: np.ndarray,
    filepath: Union[str, Path],
    extrude_depth: int = 4,
    voxel_size: float = 1.0,
):
    """
    Convert a 2-D binary array (H, W) → thin 3-D voxel slab → STL.

    Uses the same surface-triangulation approach as
    ``utils/mesh_utils.voxel_3D_binary_array_to_stl`` when available,
    otherwise falls back to trimesh voxel utilities.
    """
    H, W = binary_2d.shape
    # Extrude along z
    voxel_3d = np.repeat(binary_2d[:, :, np.newaxis], extrude_depth, axis=2)

    filepath = Path(filepath)

    # trimesh is an OPTIONAL extra: nothing in the training or evolution pipeline
    # calls this, it is here so a generated design can be exported for printing.
    try:
        import trimesh
        vg = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(voxel_3d.astype(bool))
        )
        mesh = vg.marching_cubes
        mesh.apply_scale(voxel_size)
        mesh.export(str(filepath))
        return
    except Exception:
        pass

    print(f"  [warn] Could not export STL to {filepath} (no mesh backend)")


# ---------------------------------------------------------------------------
#  CLI entry point (quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DesignGenerator quick test")
    parser.add_argument("--checkpoint", type=str,
                        default=str(_CFM_DIR / "checkpoint_0300000.pt"))
    parser.add_argument("--bc_npz", type=str,
                        default=str(_CFM_DIR / "data_00000.npz"))
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--blend", type=str, default="1,0,0,0",
                        help="Comma-separated T,F,G,L blend weights")
    parser.add_argument("--n_ode_steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="validation")
    parser.add_argument("--save_mesh", action="store_true")
    args = parser.parse_args()

    blend = tuple(float(x) for x in args.blend.split(","))
    assert len(blend) == 4, "Blend must have 4 values"

    gen = DesignGenerator(
        checkpoint_path=args.checkpoint,
        bc_npz_path=args.bc_npz,
    )

    results, dirs = gen.generate_and_save(
        n_samples=args.n_samples,
        blend=blend,
        n_ode_steps=args.n_ode_steps,
        output_root=args.output,
        save_mesh=args.save_mesh,
    )

    print(f"\nGenerated {len(results)} sample(s)")
    for r in results:
        g = r["geometry"]
        print(f"  geometry: shape={g.shape}  min={g.min():.4f}  max={g.max():.4f}  "
              f"vol={g.mean():.3f}")
