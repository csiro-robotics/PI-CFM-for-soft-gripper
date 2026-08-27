# Checkpoints

Not tracked in git (see `.gitignore`). Two are needed:

| file | produced by | used by |
|---|---|---|
| `design_encoder/encoder_best.pt` | `training/run_training.sh encoder` | stage 2 of training |
| `picfm/<run_name>/ft_XXXXXXX.pt`  | `training/run_training.sh dit`     | `evolution/` — decodes a genome into a design |

To train from scratch, run `training/run_training.sh`.
To reproduce the paper's designs without retraining, download the released
checkpoint and place it at `checkpoints/picfm/`.
