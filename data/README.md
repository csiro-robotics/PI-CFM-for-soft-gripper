# Datasets

Not tracked in git. Three condition datasets are expected:

```
data/mechanism_dataset/   # topopt
data/finray_dataset/      # finray
data/graph_dataset/       # graph
```

Each holds `data_XXXXX.npz` with a `geometry` field (128 x 64) plus the
boundary-condition channels the encoder consumes. Paths are set in
`training/configs/train_dit.yaml` and can be overridden with `DATA_ROOT`.
