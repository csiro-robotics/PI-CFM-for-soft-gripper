"""
Dataset loader for mechanism topology optimization data.

Loads npz files for conditional generation.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Union
from einops import rearrange


class MechanismConditionedDataset(Dataset):
    """
    Dataset that returns (geometry, condition) pairs for conditional generation.

    Condition channels (10 total):
        0  BCx          Fixed BC mask (x-direction)
        1  BCy          Fixed BC mask (y-direction)
        2  inputx       Prescribed displacement (x)
        3  inputy       Prescribed displacement (y)
        4  outputx      Output DOF indicator (x)
        5  outputy      Output DOF indicator (y)
        6  w_topopt     Design-space blend weight
        7  w_finray     Design-space blend weight
        8  w_graph      Design-space blend weight
        9  w_lattice    Design-space blend weight
    """

    DESIGN_SPACES = ['topopt', 'finray', 'graph', 'lattice']
    DS_CHANNEL_OFFSET = 6
    N_COND_CHANNELS = 6 + len(DESIGN_SPACES)  # 10

    def __init__(
        self,
        data_dir: Union[str, Path],
        return_img: bool = True,
        use_double: bool = False,
        normalize_geometry: bool = True,
        file_pattern: str = "data_*.npz",
        max_samples: Optional[int] = None,
        design_space_override: Optional[str] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.return_img = return_img
        self.dtype = torch.float64 if use_double else torch.float32
        self.normalize_geometry = normalize_geometry

        # Infer design space from directory name if not explicitly set
        if design_space_override is None:
            dir_name = self.data_dir.name.lower()
            for ds in self.DESIGN_SPACES:
                if ds in dir_name:
                    design_space_override = ds
                    break
        self.design_space_override = design_space_override

        # Find files
        self.file_paths = sorted(
            self.data_dir.glob(file_pattern),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        if max_samples:
            self.file_paths = self.file_paths[:max_samples]

        if not self.file_paths:
            raise FileNotFoundError(f"No npz files in {self.data_dir}")

        # Get dimensions
        sample = np.load(self.file_paths[0])
        self.height, self.width = sample['geometry'].shape

        ds_label = self.design_space_override or 'auto'
        print(f"[Dataset] {len(self.file_paths)} samples from {self.data_dir}")
        print(f"  Geometry: (1, {self.height}, {self.width})  design_space={ds_label}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        data = np.load(self.file_paths[index])

        # Geometry (target)
        geom = data['geometry'].astype(np.float32)
        if self.normalize_geometry:
            geom = geom * 2.0 - 1.0
        geom = torch.tensor(geom, dtype=self.dtype).unsqueeze(0)

        vol_frac = float(data.get('volume_fraction', 0.3))

        # Condition: 6 BC channels
        bc_channels = ['BCx', 'BCy', 'inputx', 'inputy', 'outputx', 'outputy']
        cond = np.stack([data[ch].astype(np.float32) for ch in bc_channels], axis=0)
        cond = torch.tensor(cond, dtype=self.dtype)

        H, W = cond.shape[1], cond.shape[2]

        # Channels 6-9: design-space blend weights (broadcast across H×W)
        ds_channels = torch.zeros(len(self.DESIGN_SPACES), H, W, dtype=self.dtype)
        if self.design_space_override is not None:
            ds_name = self.design_space_override
        else:
            ds_name = str(data.get('design_space', 'topopt'))
        if ds_name in self.DESIGN_SPACES:
            ds_idx = self.DESIGN_SPACES.index(ds_name)
        else:
            ds_idx = 0
        ds_channels[ds_idx, :, :] = 1.0

        cond = torch.cat([cond, ds_channels], dim=0)  # (10, H, W)

        if not self.return_img:
            geom = rearrange(geom, 'c h w -> (h w) c')
            cond = rearrange(cond, 'c h w -> (h w) c')

        return geom, cond, vol_frac

    @property
    def geometry_shape(self) -> Tuple[int, int, int]:
        return (1, self.height, self.width)

    @property
    def condition_shape(self) -> Tuple[int, int, int]:
        return (self.N_COND_CHANNELS, self.height, self.width)
