# Checkpoints

Not tracked in git (see `.gitignore`). Two are needed:

| file | produced by | used by |
|---|---|---|
| `design_encoder/encoder_best.pt` | `training/run_training.sh encoder` | stage 2 of training |
| `picfm_dit.pt` | `training/run_training.sh dit` | `evolution/` — decodes a genome into a design |

Stage 2 writes its checkpoints into `picfm/<run_name>/ft_XXXXXXX.pt`. Copy or
symlink the one you want the search to use to `checkpoints/picfm_dit.pt`, which is
where `evolution/` looks by default:

```bash
ln -s picfm/<run_name>/ft_0066000.pt checkpoints/picfm_dit.pt
```

Or point at it explicitly, without the symlink:

```bash
CHECKPOINT=checkpoints/picfm/<run_name>/ft_0066000.pt evolution/run_evolution.sh
```

To reproduce the paper's designs without retraining, download the released
checkpoint and place it at `checkpoints/picfm_dit.pt`.
