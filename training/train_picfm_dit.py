"""
PICFM-DiT Training Script.

Single-stage training of a Lumina-architecture DiT with SpatialStyleEncoder v2
for conditional flow matching on topology optimization.

  Model:    DiT (Lumina-T2I) + SpatialStyleEncoder → context tokens
  Flow:     Guided OT-CFM (label-preserving optimal transport)
  Physics:  Batched penalty-method FEM with PIDM noise-aware weighting
  CFG:      Learned null tokens with per-sample dropout

Usage:
    python train_picfm_DiT.py --config configs/train_dit.yaml
    python train_picfm_DiT.py --config configs/train_dit.yaml --batch_size 64
"""

import os
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from flows.DiT import DiTModel, DiT_S_4, DiT_B_4, DiT_S_8
from src.training_utils_dit import (
    EMA,
    warmup_lr,
    euler_sample_dit,
    visualize_samples_dit,
    visualize_label_comparison_dit,
)
from src.physics_setup import setup_physics_from_condition
from flows.residual_physics import compute_batched_physics_loss
from src.mechanism_dataset import (
    MechanismConditionedDataset,
    build_multi_dataset,
    cycle,
)
from fem_solver.torch_fem_solver_fast import TorchFEMSolverFast


# ======================================================================
# Config
# ======================================================================

_YAML_KEY_MAP = {
    "data.data_dirs":              "data_dirs",
    "data.data_dir":               "data_dir",
    "data.max_samples":            "max_samples",
    "model.hidden_dim":            "hidden_dim",
    "model.depth":                 "depth",
    "model.num_heads":             "num_heads",
    "model.patch_size":            "patch_size",
    "model.mlp_ratio":             "mlp_ratio",
    "model.num_ds":                "num_ds",
    "model.bc_channels":           "bc_channels",
    "model.ds_channel_offset":     "ds_channel_offset",
    "flow_matching.sigma":         "fm_sigma",
    "flow_matching.type":          "fm_type",
    "flow_matching.timestep_sampling": "timestep_sampling",
    "physics.lambda_physics":      "lambda_physics",
    "physics.physics_start_iter":  "physics_start_iter",
    "physics.physics_ramp_iters":  "physics_ramp_iters",
    "physics.physics_min_t":       "physics_min_t",
    "physics.physics_every":       "physics_every",
    "physics.w_mechanism":         "w_mechanism",
    "physics.w_binary":            "w_binary",
    "physics.penal":               "penal",
    "physics.rmin":                "rmin",
    "physics.E_min":               "E_min",
    "training.batch_size":         "batch_size",
    "training.lr":                 "lr",
    "training.grad_clip":          "grad_clip",
    "training.total_steps":        "total_steps",
    "training.warmup":             "warmup",
    "training.ema_decay":          "ema_decay",
    "training.ema_start":          "ema_start",
    "training.cond_drop_prob":     "cond_drop_prob",
    "training.cfg_scale":          "cfg_scale",
    "training.pretrained_encoder": "pretrained_encoder",
    "training.freeze_encoder":     "freeze_encoder",
    "training.encoder_lr_scale":   "encoder_lr_scale",
    "logging.log_freq":            "log_freq",
    "logging.eval_freq":           "eval_freq",
    "logging.sample_freq":         "sample_freq",
    "logging.save_freq":           "save_freq",
    "logging.n_ode_steps":         "n_ode_steps",
    "logging.n_sample_vis":        "n_sample_vis",
    "output.output_dir":           "output_dir",
    "output.run_name":             "run_name",
    "output.resume":               "resume",
    "output.num_workers":          "num_workers",
    "output.wandb":                "wandb",
}

_DEFAULTS = {
    "data_dirs": None, "data_dir": "data/mechanism_dataset", "max_samples": None,
    "hidden_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 4,
    "mlp_ratio": 4.0, "num_ds": 4, "bc_channels": 6, "ds_channel_offset": 6,
    "fm_sigma": 0.0, "fm_type": "otcfm", "timestep_sampling": "uniform",
    "lambda_physics": 0.1, "physics_start_iter": 500,
    "physics_ramp_iters": 10000, "physics_min_t": 0.5,
    "physics_every": 1,
    "w_mechanism": 1.0, "w_binary": 1.0,
    "penal": 5.0, "rmin": 0.5, "E_min": 1e-4,
    "batch_size": 64, "lr": 1e-4, "grad_clip": 1.0,
    "total_steps": 200000, "warmup": 2000,
    "ema_decay": 0.9999, "ema_start": 1000,
    "cond_drop_prob": 0.1, "cfg_scale": 3.0,
    "pretrained_encoder": None, "freeze_encoder": True,
    "encoder_lr_scale": 0.1,
    "log_freq": 20, "eval_freq": 500,
    "sample_freq": 5000, "save_freq": 10000,
    "n_ode_steps": 100, "n_sample_vis": 4,
    "output_dir": "./trained_models/picfm", "run_name": "run_dit",
    "resume": None, "num_workers": 4, "wandb": False,
}


def _flatten_yaml(cfg: dict, parent: str = "") -> dict:
    items = {}
    for k, v in cfg.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten_yaml(v, key))
        else:
            items[key] = v
    return items


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_dit.yaml")

    # All scalar CLI overrides
    _CLI = {
        "data_dir": str, "max_samples": int,
        "hidden_dim": int, "depth": int, "num_heads": int, "patch_size": int,
        "mlp_ratio": float, "num_ds": int, "bc_channels": int, "ds_channel_offset": int,
        "fm_sigma": float, "fm_type": str, "timestep_sampling": str,
        "lambda_physics": float, "physics_start_iter": int,
        "physics_ramp_iters": int, "physics_min_t": float, "physics_every": int,
        "w_mechanism": float, "w_binary": float,
        "penal": float, "rmin": float, "E_min": float,
        "batch_size": int, "lr": float, "grad_clip": float,
        "total_steps": int, "warmup": int,
        "ema_decay": float, "ema_start": int,
        "cond_drop_prob": float, "cfg_scale": float,
        "pretrained_encoder": str, "encoder_lr_scale": float,
        "log_freq": int, "eval_freq": int, "sample_freq": int, "save_freq": int,
        "n_ode_steps": int, "n_sample_vis": int,
        "output_dir": str, "run_name": str, "resume": str, "num_workers": int,
    }
    for name, tp in _CLI.items():
        parser.add_argument(f"--{name}", type=tp, default=None)
    parser.add_argument("--data_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--freeze_encoder", action="store_true", default=None)

    cli = parser.parse_args()

    cfg_path = Path(cli.config)
    yaml_cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        print(f"[Config] Loaded {cfg_path}")
    flat = _flatten_yaml(yaml_cfg)

    merged = {}
    for flat_key, attr in _YAML_KEY_MAP.items():
        cli_val = getattr(cli, attr, None)
        yaml_val = flat.get(flat_key)
        if cli_val is not None:
            merged[attr] = cli_val
        elif yaml_val is not None:
            merged[attr] = yaml_val
        else:
            merged[attr] = _DEFAULTS.get(attr)
    merged["config"] = str(cfg_path)
    return argparse.Namespace(**merged)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.output_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "samples").mkdir(exist_ok=True)

    with open(save_dir / "args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    # W&B
    log_fn = lambda d, step: None
    if args.wandb:
        import wandb
        wandb.init(project="picfm_topop", name=args.run_name, config=vars(args))
        log_fn = lambda d, step: wandb.log(d, step=step)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("\n--- Dataset ---")
    if args.data_dirs is not None:
        ds_train = build_multi_dataset(
            args.data_dirs, max_samples=args.max_samples,
            return_img=True, normalize_geometry=False,
        )
        geom_sample, cond_sample, _ = ds_train.datasets[0][0]
    else:
        ds_train = MechanismConditionedDataset(
            data_dir=args.data_dir, return_img=True,
            normalize_geometry=False, max_samples=args.max_samples,
        )
        geom_sample, cond_sample, _ = ds_train[0]

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    dl_iter = cycle(dl_train)

    _, H, W = geom_sample.shape
    cond_channels = cond_sample.shape[0]
    print(f"  Image: {H}x{W}  Cond channels: {cond_channels}  Samples: {len(ds_train)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n--- Model ---")
    model = DiTModel(
        img_height=H, img_width=W,
        patch_size=args.patch_size,
        in_channels=1, out_channels=1,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        cond_spatial_channels=cond_channels,
        num_ds=args.num_ds,
        bc_channels=args.bc_channels,
        ds_channel_offset=args.ds_channel_offset,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = (H // args.patch_size) * (W // args.patch_size)
    print(f"  DiT: D={args.hidden_dim}, depth={args.depth}, heads={args.num_heads}, patch={args.patch_size}")
    print(f"  Params: {n_params:,}  Tokens: {n_tokens}")

    # ------------------------------------------------------------------
    # Pre-trained encoder
    # ------------------------------------------------------------------
    pretrained_enc = args.pretrained_encoder
    freeze_enc = args.freeze_encoder
    enc_lr_scale = args.encoder_lr_scale

    if pretrained_enc:
        print(f"\n--- Encoder ---")
        enc_ckpt = torch.load(pretrained_enc, map_location=device, weights_only=False)
        missing, unexpected = model.cond_encoder.load_state_dict(
            enc_ckpt["encoder_state_dict"], strict=False,
        )
        if missing:
            print(f"  Missing keys (expected for new model): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Loaded encoder from {pretrained_enc}")

        if freeze_enc:
            for p in model.cond_encoder.parameters():
                p.requires_grad_(False)
            print(f"  Encoder frozen")
        else:
            print(f"  Encoder trainable (LR scale={enc_lr_scale})")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable:,}")

    # ------------------------------------------------------------------
    # EMA + Optimizer
    # ------------------------------------------------------------------
    ema = EMA(
        decay=args.ema_decay,
        exclude_prefix='cond_encoder.' if freeze_enc else None,
    )
    ema.register(model)

    if pretrained_enc and not freeze_enc:
        enc_params = list(model.cond_encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        dit_params = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in enc_ids]
        optimizer = optim.AdamW([
            {"params": dit_params, "lr": args.lr},
            {"params": enc_params, "lr": args.lr * enc_lr_scale},
        ], weight_decay=0.0)
    else:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.0,
        )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_lr(step, args.warmup),
    )

    # ------------------------------------------------------------------
    # Flow Matching
    # ------------------------------------------------------------------
    if args.fm_type == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.fm_sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=args.fm_sigma)
    print(f"\n--- Flow Matching: {args.fm_type} (sigma={args.fm_sigma}) ---")
    print(f"  Timestep sampling: {args.timestep_sampling}")

    # ------------------------------------------------------------------
    # Physics solver
    # ------------------------------------------------------------------
    base_solver = None
    use_physics = args.lambda_physics > 0
    if use_physics:
        print(f"\n--- Physics ---")
        dummy_fixed = torch.tensor([0, 1], dtype=torch.long, device=device)
        base_solver = TorchFEMSolverFast(
            nx=W, ny=H, fixed_dofs=dummy_fixed,
            penal=args.penal, rmin=args.rmin, E_min=args.E_min,
            device=str(device), dtype=torch.float32,
        )
        print(f"  Grid: {W}x{H}  penal={args.penal}, rmin={args.rmin}")
        print(f"  lambda={args.lambda_physics}, start={args.physics_start_iter}, "
              f"ramp={args.physics_ramp_iters}, min_t={args.physics_min_t}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_iter = 0
    if args.resume:
        print(f"\n--- Resuming from {args.resume} ---")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]
        start_iter = ckpt.get("iteration", 0)
        print(f"  Resumed at iteration {start_iter}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training for {args.total_steps} iterations")
    print(f"{'='*60}\n")

    pbar = tqdm(range(start_iter, args.total_steps + 1), initial=start_iter,
                total=args.total_steps + 1, dynamic_ncols=True)

    loss_cfm_ema = None
    loss_phys_ema = None
    sub_loss_ema = {}
    last_good_ckpt = None

    for iteration in pbar:
        model.train()

        # ---- Load batch ----
        geom, cond, _ = next(dl_iter)
        geom = geom.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        x_1 = geom
        x_0 = torch.randn_like(x_1)

        # ---- CFG dropout mask ----
        null_mask = None
        if args.cond_drop_prob > 0 and iteration >= args.warmup:
            null_mask = torch.rand(x_1.shape[0], device=device) < args.cond_drop_prob

        # ---- Timestep sampling ----
        if args.timestep_sampling == "lognormal":
            t_pre = torch.sigmoid(torch.randn(x_0.shape[0], device=device))
        else:
            t_pre = None

        # ---- Guided OT-CFM (label-preserving) ----
        if args.fm_type == "otcfm":
            batch_idx = torch.arange(x_0.shape[0], device=device)
            t, x_t, u_t, y0_idx, y1_idx = FM.guided_sample_location_and_conditional_flow(
                x_0, x_1, y0=batch_idx, y1=batch_idx, t=t_pre,
            )
            cond = cond[y1_idx.long()]
            x_0 = x_0[y0_idx.long()]
            if null_mask is not None:
                null_mask = null_mask[y1_idx.long()]
        else:
            t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1, t=t_pre)

        # ---- Forward ----
        v_t = model(t.to(device), x_t, cond_spatial=cond, null_mask=null_mask)

        # ---- CFM loss ----
        loss_cfm = (v_t - u_t).pow(2).mean()

        # NaN check
        if not torch.isfinite(loss_cfm):
            tqdm.write(f"  [NaN iter {iteration}] Skipping")
            continue

        # ---- Physics loss ----
        loss_phys = torch.tensor(0.0, device=device)
        phys_weight = 0.0
        phys_ok = False
        sub_losses = {}
        compute_phys = (
            use_physics
            and iteration >= args.physics_start_iter
            and iteration % args.physics_every == 0
        )

        if compute_phys:
            try:
                x_1_hat = x_0 + v_t
                rho = x_1_hat[:, 0].clamp(0, 1).transpose(1, 2)

                t_mask = t >= args.physics_min_t
                if t_mask.any():
                    t_valid = t[t_mask]
                    sigma_phys = (1.0 - t_valid).clamp(min=0.01).to(device)
                    valid_rho = rho[t_mask].reshape(t_mask.sum(), -1)
                    valid_cond = cond[t_mask]

                    out = compute_batched_physics_loss(
                        valid_rho, valid_cond, base_solver,
                        sigma=sigma_phys,
                        w_mechanism=args.w_mechanism,
                        w_binary=args.w_binary,
                    )

                    if torch.isfinite(out["total"]):
                        t_weight = t_valid.pow(1).mean()
                        loss_phys = out["total"] * t_weight
                        phys_ok = True
                        sub_losses = {k: out[k].item() for k in
                                      ["mechanism", "binary"]}
            except Exception as e:
                if iteration % 500 == 0:
                    tqdm.write(f"  [Physics err iter {iteration}] {e}")
                loss_phys = torch.tensor(0.0, device=device)
                phys_ok = False

        # L_mech = sign * R_out is NEGATIVE for a good mechanism, so the weight
        # must gate on "physics was computed", never on the sign of the loss.
        if phys_ok and torch.isfinite(loss_phys):
            progress = (iteration - args.physics_start_iter) / max(args.physics_ramp_iters, 1)
            phys_weight = args.lambda_physics * min(progress, 1.0)
        else:
            phys_weight = 0.0

        # ---- Total loss ----
        loss = loss_cfm + phys_weight * loss_phys

        # ---- Backward ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient NaN check
        grad_ok = True
        for p in model.parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                grad_ok = False
                break
        if not grad_ok:
            tqdm.write(f"  [Grad NaN iter {iteration}] Skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration > args.ema_start:
            ema.update(model)

        # ---- In-memory checkpoint ----
        if iteration % 1000 == 0:
            last_good_ckpt = {
                "iteration": iteration,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_shadow": {k: v.cpu().clone() for k, v in ema.shadow.items()} if ema.shadow else None,
            }

        # ---- Logging ----
        cfm_val = loss_cfm.item()
        loss_cfm_ema = cfm_val if loss_cfm_ema is None else 0.95 * loss_cfm_ema + 0.05 * cfm_val
        if sub_losses:
            for k, v in sub_losses.items():
                loss_phys_ema = v if loss_phys_ema is None else 0.95 * loss_phys_ema + 0.05 * v
                sub_loss_ema[k] = v if k not in sub_loss_ema else 0.95 * sub_loss_ema[k] + 0.05 * v

        if iteration % args.log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            phys_str = f"phys={loss_phys_ema:.2e}" if loss_phys_ema is not None else "phys=-"
            pbar.set_description(f"cfm={loss_cfm_ema:.3e} {phys_str} w={phys_weight:.2e} lr={lr_now:.1e}")

            log_dict = {
                "loss_cfm": cfm_val,
                "loss_physics": loss_phys.item() if compute_phys else 0.0,
                "loss_total": loss.item(),
                "physics_weight": phys_weight,
                "lr": lr_now,
            }
            for k, v in sub_loss_ema.items():
                log_dict[f"sub/{k}"] = v
            log_fn(log_dict, step=iteration)

        if iteration % 500 == 0 and iteration > 0:
            tqdm.write(f"  [Iter {iteration}] cfm={cfm_val:.3e} phys={loss_phys.item():.3e}")

        # ---- Validation ----
        if iteration % args.eval_freq == 0 and iteration > 0:
            model.eval()
            ema.apply_shadow(model)
            with torch.no_grad():
                geom_v, cond_v, _ = next(dl_iter)
                geom_v = geom_v.to(device)
                cond_v = cond_v.to(device)
                x_0_v = torch.randn_like(geom_v)
                if args.fm_type == "otcfm":
                    idx_v = torch.arange(x_0_v.shape[0], device=device)
                    t_v, x_t_v, u_t_v, _, y1v = FM.guided_sample_location_and_conditional_flow(
                        x_0_v, geom_v, y1=idx_v,
                    )
                    cond_v = cond_v[y1v.long()]
                else:
                    t_v, x_t_v, u_t_v = FM.sample_location_and_conditional_flow(x_0_v, geom_v)
                v_v = model(t_v.to(device), x_t_v, cond_spatial=cond_v)
                val_loss = (v_v - u_t_v).pow(2).mean().item()
            tqdm.write(f"  [Val {iteration}] cfm={val_loss:.3e}")
            log_fn({"loss_val_cfm": val_loss}, step=iteration)
            ema.restore(model)

        # ---- Sample ----
        if iteration % args.sample_freq == 0 and iteration > 0:
            tqdm.write(f"  Sampling at iter {iteration}...")
            model.eval()
            ema.apply_shadow(model)

            geom_s, cond_s, _ = next(dl_iter)
            n_vis = min(args.n_sample_vis, geom_s.shape[0])
            geom_s = geom_s[:n_vis].to(device)
            cond_s = cond_s[:n_vis].to(device)

            with torch.no_grad():
                samples = euler_sample_dit(
                    model, cond_s, shape=(n_vis, 1, H, W),
                    n_steps=args.n_ode_steps, device=str(device),
                    cfg_scale=args.cfg_scale,
                )

            vis_path = save_dir / "samples" / f"iter_{iteration:07d}.png"
            visualize_samples_dit(samples, geom_s, cond_s, str(vis_path), n_show=n_vis)
            tqdm.write(f"  Saved: {vis_path}")

            # Label comparison: same BCs, different DS labels
            cmp_path = save_dir / "samples" / f"label_cmp_{iteration:07d}.png"
            visualize_label_comparison_dit(
                model, cond_s[0].cpu(), str(cmp_path),
                n_ode_steps=args.n_ode_steps, cfg_scale=args.cfg_scale,
                device=str(device),
            )

            # Physics metrics on generated samples
            if use_physics:
                try:
                    rho_gen = samples[:, 0].clamp(0, 1).transpose(1, 2).reshape(n_vis, -1)
                    engine = setup_physics_from_condition(
                        cond_s[0], base_solver, device=str(device),
                        w_mechanism=args.w_mechanism,
                        w_binary=args.w_binary,
                    )
                    metrics = engine.compute_metrics(rho_gen[0])
                    tqdm.write(
                        f"  Physics: bin={metrics['binary_score']:.3f} R={metrics['reaction_force']:.4f}"
                    )
                except Exception as e:
                    tqdm.write(f"  [Physics metrics err] {e}")

            ema.restore(model)

        # ---- Checkpoint ----
        if iteration % args.save_freq == 0 and iteration > 0:
            ckpt_path = save_dir / f"checkpoint_{iteration:07d}.pt"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_shadow": ema.shadow,
                "args": vars(args),
            }, str(ckpt_path))
            tqdm.write(f"  Saved: {ckpt_path}")

    # Final save
    final_path = save_dir / "model_final.pt"
    torch.save({
        "iteration": args.total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "args": vars(args),
    }, str(final_path))
    print(f"\nDone. Final model: {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
