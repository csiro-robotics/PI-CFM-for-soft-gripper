"""
Configuration management for PICFM training.

Handles YAML loading, CLI argument parsing, and config merging.
Priority: CLI flags > YAML file > built-in defaults.
"""

import argparse
from pathlib import Path
import yaml


# -- YAML → flat namespace mapping ----------------------------------------
# Nested YAML keys are flattened with the following aliases so that the
# rest of the training code keeps using  args.batch_size, args.lr, etc.
_YAML_KEY_MAP = {
    # data
    "data.data_dirs":             "data_dirs",
    "data.data_dir":              "data_dir",
    "data.max_samples":           "max_samples",
    # model
    "model.model_channels":       "model_channels",
    "model.channel_mult":         "channel_mult",
    "model.num_res_blocks":       "num_res_blocks",
    "model.attention_resolutions": "attention_resolutions",
    "model.dropout":              "dropout",
    # flow_matching
    "flow_matching.sigma":        "fm_sigma",
    "flow_matching.type":         "fm_type",
    # physics
    "physics.lambda_physics":     "lambda_physics",
    "physics.physics_start_iter": "physics_start_iter",
    "physics.physics_ramp_iters": "physics_ramp_iters",
    "physics.physics_min_t":      "physics_min_t",
    "physics.physics_every":      "physics_every",
    "physics.physics_batch_size": "physics_batch_size",
    "physics.w_mechanism":        "w_mechanism",
    "physics.w_binary":           "w_binary",
    "physics.w_compliance":       "w_compliance",
    "physics.w_tv":               "w_tv",
    "physics.penal":              "penal",
    "physics.rmin":               "rmin",
    "physics.E_min":              "E_min",
    # training
    "training.batch_size":        "batch_size",
    "training.lr":                "lr",
    "training.grad_clip":         "grad_clip",
    "training.total_steps":       "total_steps",
    "training.warmup":            "warmup",
    "training.ema_decay":         "ema_decay",
    "training.ema_start":         "ema_start",
    "training.cond_drop_prob":    "cond_drop_prob",
    "training.cfg_scale":         "cfg_scale",
    "training.lambda_ortho":      "lambda_ortho",
    # logging
    "logging.log_freq":           "log_freq",
    "logging.eval_freq":          "eval_freq",
    "logging.sample_freq":        "sample_freq",
    "logging.save_freq":          "save_freq",
    "logging.n_ode_steps":        "n_ode_steps",
    "logging.n_sample_vis":       "n_sample_vis",
    # output
    "output.output_dir":          "output_dir",
    "output.run_name":            "run_name",
    "output.resume":              "resume",
    "output.finetune":            "finetune",
    "output.num_workers":         "num_workers",
    "output.wandb":               "wandb",
}

# Hardcoded defaults (same values as the old argparse defaults)
_DEFAULTS = {
    "data_dirs": None, "data_dir": "data/mechanism_dataset", "max_samples": None,
    "model_channels": 64, "channel_mult": [1, 2, 3, 4],
    "num_res_blocks": 2, "attention_resolutions": "16,8", "dropout": 0.0,
    "fm_sigma": 0.0, "fm_type": "otcfm",
    "lambda_physics": 0.1, "physics_start_iter": 5000,
    "physics_ramp_iters": 10000, "physics_min_t": 0.5,
    "physics_every": 1, "physics_batch_size": 4,
    "w_mechanism": 1.0, "w_binary": 1.0,
    "w_compliance": 0.1, "w_tv": 0.05, "penal": 5.0, "rmin": 0.5, "E_min": 1e-4,
    "batch_size": 8, "lr": 2e-4, "grad_clip": 1.0,
    "total_steps": 300000, "warmup": 5000,
    "ema_decay": 0.9999, "ema_start": 1000,
    "cond_drop_prob": 0.0, "cfg_scale": 1.0, "lambda_ortho": 0.1,
    "log_freq": 20, "eval_freq": 500,
    "sample_freq": 10000, "save_freq": 10000,
    "n_ode_steps": 100, "n_sample_vis": 4,
    "output_dir": "./trained_models/picfm", "run_name": "run_1",
    "resume": None, "finetune": None, "num_workers": 4, "wandb": False,
}


def _flatten_yaml(cfg: dict, parent: str = "") -> dict:
    """Recursively flatten nested dict into dot-separated keys."""
    items = {}
    for k, v in cfg.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten_yaml(v, key))
        else:
            items[key] = v
    return items


def build_resolved_yaml(args: argparse.Namespace) -> dict:
    """Build a self-contained YAML snapshot from resolved args."""
    return {
        "data":          {k: getattr(args, k) for k in ["data_dirs", "data_dir", "max_samples"]},
        "model":         {k: getattr(args, k) for k in ["model_channels", "channel_mult",
                          "num_res_blocks", "attention_resolutions", "dropout"]},
        "flow_matching": {"sigma": args.fm_sigma, "type": args.fm_type},
        "physics":       {k: getattr(args, k) for k in ["lambda_physics", "physics_start_iter",
                          "physics_ramp_iters", "physics_min_t", "physics_every",
                          "physics_batch_size", "w_mechanism", "w_binary",
                          "w_compliance", "w_tv", "penal", "rmin", "E_min"]},
        "training":      {k: getattr(args, k) for k in ["batch_size", "lr", "grad_clip",
                          "total_steps", "warmup", "ema_decay", "ema_start",
                          "cond_drop_prob", "cfg_scale", "lambda_ortho"]},
        "logging":       {k: getattr(args, k) for k in ["log_freq", "eval_freq",
                          "sample_freq", "save_freq", "n_ode_steps", "n_sample_vis"]},
        "output":        {k: getattr(args, k) for k in ["output_dir", "run_name",
                          "resume", "finetune", "num_workers", "wandb"]},
    }


def parse_args() -> argparse.Namespace:
    """Load configuration from YAML file, then override with any CLI flags.

    Priority:  CLI flag  >  YAML file  >  built-in defaults

    Usage:
        python train_picfm.py --config configs/default.yaml
        python train_picfm.py --config configs/default.yaml --lr 1e-5 --batch_size 32
    """
    parser = argparse.ArgumentParser(
        description="PICFM Training for Mechanism Design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file")

    # Every flat key is also a CLI flag so users can override any value.
    _CLI_TYPES = {
        "data_dir": str, "max_samples": int,
        "model_channels": int, "num_res_blocks": int,
        "attention_resolutions": str, "dropout": float,
        "fm_sigma": float, "fm_type": str,
        "lambda_physics": float, "physics_start_iter": int,
        "physics_ramp_iters": int, "physics_min_t": float,
        "physics_every": int, "physics_batch_size": int,
        "w_mechanism": float,
        "w_binary": float, "w_compliance": float, "w_tv": float,
        "penal": float, "rmin": float, "E_min": float,
        "batch_size": int, "lr": float, "grad_clip": float,
        "total_steps": int, "warmup": int,
        "ema_decay": float, "ema_start": int,
        "cond_drop_prob": float, "cfg_scale": float,
        "lambda_ortho": float,
        "log_freq": int, "eval_freq": int,
        "sample_freq": int, "save_freq": int,
        "n_ode_steps": int, "n_sample_vis": int,
        "output_dir": str, "run_name": str,
        "resume": str, "num_workers": int,
        "finetune": str,
    }
    for name, tp in _CLI_TYPES.items():
        parser.add_argument(f"--{name}", type=tp, default=None)
    # data_dirs: multi-dataset (list of dir:label pairs)
    parser.add_argument("--data_dirs", type=str, nargs="+", default=None,
                        help="List of dir:label pairs, e.g. data/mechanism_dataset:topopt data/finray_dataset:finray")
    # channel_mult needs nargs
    parser.add_argument("--channel_mult", type=int, nargs="+", default=None)
    # wandb is a flag
    parser.add_argument("--wandb", action="store_true", default=None)

    cli_args = parser.parse_args()

    # -- Load YAML --
    cfg_path = Path(cli_args.config)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        print(f"[Config] Loaded {cfg_path}")
    else:
        yaml_cfg = {}
        print(f"[Config] No YAML found at {cfg_path}, using CLI / defaults only")

    flat_yaml = _flatten_yaml(yaml_cfg)

    # -- Merge (CLI > YAML > hardcoded defaults) --
    merged = {}
    for flat_key, attr_name in _YAML_KEY_MAP.items():
        cli_val = getattr(cli_args, attr_name, None)
        yaml_val = flat_yaml.get(flat_key)
        if cli_val is not None:
            merged[attr_name] = cli_val
        elif yaml_val is not None:
            merged[attr_name] = yaml_val
        else:
            merged[attr_name] = _DEFAULTS[attr_name]

    merged["config"] = str(cfg_path)

    args = argparse.Namespace(**merged)
    return args
