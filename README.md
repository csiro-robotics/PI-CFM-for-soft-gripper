# PI-CFM for Soft Grippers

Physics-Informed Conditional Flow Matching for the generative design of soft
robotic gripper fingers, with quality-diversity search over the generative
latent and grasp evaluation by a differentiable-contact FEM solver.

The repository is **self-contained**: training, generation, evolution and
simulation all live here, with no external project dependencies.

```
training/     stage 1 condition encoder -> stage 2 PI-CFM DiT
evolution/    Voronoi+token genome -> CFM -> design -> implicit FEM grasp -> 4-D CVT MAP-Elites
```

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

A CUDA GPU is required: the FEM solver runs in NVIDIA Warp and the DiT trains on GPU.
Tested on Python 3.12 / CUDA 12.4 / RTX A5000 and H100.

## 2. Data and checkpoints

Neither is tracked in git — see [`data/README.md`](data/README.md) and
[`checkpoints/README.md`](checkpoints/README.md).

```
data/mechanism_dataset/   (topopt)
data/finray_dataset/      (finray)
data/graph_dataset/       (graph)
```

## 3. Training

Two stages, **in order** — the DiT is trained against a frozen, pre-trained
condition encoder, so stage 2 without stage 1 trains against random features.

```bash
cd training
./run_training.sh              # both stages
./run_training.sh encoder      # stage 1 only  -> checkpoints/design_encoder/encoder_best.pt
./run_training.sh dit          # stage 2 only  -> checkpoints/picfm/<run_name>/
```

| stage | script | what it trains |
|---|---|---|
| 1 | `pretrain_encoder.py` | `SpatialStyleEncoder`: BC-CNN reconstruction + design-space classification + fusion |
| 2 | `train_picfm_dit.py`  | Lumina-DiT flow matching with the mechanism-FEM physics loss |

The launcher is written for a **single local desktop GPU** (no SLURM). Batch
sizes default lower than the published run; override via the environment:

```bash
DATA_ROOT=../data ENC_EPOCHS=150 DIT_BATCH=64 ./run_training.sh
```

Stage-2 hyper-parameters live in [`training/configs/train_dit.yaml`](training/configs/train_dit.yaml);
any scalar can be overridden on the command line (`--lr`, `--total_steps`, …).

## 4. Evolution

*(added in the next step)*

---

## Licence

CSIRO Open Source Software Licence Agreement (a variation of the BSD/MIT
licence) — see [`LICENSE`](LICENSE).

## Citation

*(to be added on publication)*
