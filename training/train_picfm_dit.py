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
"""
Physics-Informed Conditional Flow Matching (PICFM) Training Script — DiT Variant.

Trains a Diffusion Transformer (DiT) to generate mechanism topologies conditioned on
boundary conditions (BCs, input force, output direction). The model learns
the velocity field v(x_t, t, cond) via conditional flow matching.

This is a STANDALONE script — it does NOT modify train_picfm.py so that UNet v5
and DiT can train in parallel.

Loss = L_cfm + λ_phys * L_physics(σ)

Usage:
    python train_picfm_DiT.py --config configs/train_dit.yaml
    python train_picfm_DiT.py --config configs/train_dit.yaml --batch_size 8

Multi-dataset:
    python train_picfm_DiT.py --config configs/train_dit.yaml \
        --data_dirs data/mechanism_dataset:topopt data/finray_dataset:finray
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

# Local modules — DiT model + DiT utils
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
from ds_classifier import DSClassifierCNN


# ======================================================================
# DiT-specific config parsing
# ======================================================================
# Extends the YAML/CLI system with DiT architecture parameters.
# We duplicate the config logic here to keep this script self-contained.

_YAML_KEY_MAP = {
    # data
    "data.data_dirs":              "data_dirs",
    "data.data_dir":               "data_dir",
    "data.max_samples":            "max_samples",
    # DiT model
    "model.hidden_dim":            "hidden_dim",
    "model.depth":                 "depth",
    "model.num_heads":             "num_heads",
    "model.patch_size":            "patch_size",
    "model.mlp_ratio":             "mlp_ratio",
    "model.bc_channels":           "bc_channels",
    "model.num_design_spaces":     "num_design_spaces",
    "model.n_concept_layers":      "n_concept_layers",
    "model.n_concept_heads":       "n_concept_heads",
    "model.dropout":               "dropout",
    "model.bc_token_mode":          "bc_token_mode",
    # Also accept UNet-style keys (ignored if not present)
    "model.model_channels":        "model_channels",
    "model.channel_mult":          "channel_mult",
    "model.num_res_blocks":        "num_res_blocks",
    "model.attention_resolutions": "attention_resolutions",
    # flow_matching
    "flow_matching.sigma":         "fm_sigma",
    "flow_matching.type":          "fm_type",
    "flow_matching.timestep_sampling": "timestep_sampling",
    # physics
    "physics.lambda_physics":      "lambda_physics",
    "physics.physics_start_iter":  "physics_start_iter",
    "physics.physics_ramp_iters":  "physics_ramp_iters",
    "physics.physics_min_t":       "physics_min_t",
    "physics.physics_every":       "physics_every",
    "physics.physics_batch_size":  "physics_batch_size",
    "physics.w_mechanism":         "w_mechanism",
    "physics.w_binary":            "w_binary",
    "physics.w_compliance":        "w_compliance",
    "physics.w_tv":                "w_tv",
    "physics.penal":               "penal",
    "physics.rmin":                "rmin",
    "physics.E_min":               "E_min",
    # training
    "training.batch_size":         "batch_size",
    "training.lr":                 "lr",
    "training.grad_clip":          "grad_clip",
    "training.total_steps":        "total_steps",
    "training.warmup":             "warmup",
    "training.ema_decay":          "ema_decay",
    "training.ema_start":          "ema_start",
    "training.cond_drop_prob":     "cond_drop_prob",
    "training.cfg_scale":          "cfg_scale",
    # v9 encoder
    "training.pretrained_encoder":  "pretrained_encoder",
    "training.freeze_encoder":      "freeze_encoder",
    "training.lambda_cls":          "lambda_cls",
    "training.encoder_lr_scale":    "encoder_lr_scale",
    "training.lambda_diversity":    "lambda_diversity",
    "training.contrastive_margin":   "contrastive_margin",
    # 2-stage training
    "training.training_stage":       "training_stage",
    "training.freeze_backbone":      "freeze_backbone",
    "training.disable_ds_tokens":    "disable_ds_tokens",
    "training.stage1_checkpoint":    "stage1_checkpoint",
    # DS classifier loss (Stage 2.1)
    "training.ds_classifier_path":   "ds_classifier_path",
    "training.lambda_ds_cls":        "lambda_ds_cls",
    "training.ds_cls_min_t":         "ds_cls_min_t",
    "training.ds_cls_start_iter":    "ds_cls_start_iter",
    "training.ds_cls_ramp_iters":    "ds_cls_ramp_iters",
    # logging
    "logging.log_freq":            "log_freq",
    "logging.eval_freq":           "eval_freq",
    "logging.sample_freq":         "sample_freq",
    "logging.save_freq":           "save_freq",
    "logging.n_ode_steps":         "n_ode_steps",
    "logging.n_sample_vis":        "n_sample_vis",
    # output
    "output.output_dir":           "output_dir",
    "output.run_name":             "run_name",
    "output.resume":               "resume",
    "output.finetune":             "finetune",
    "output.num_workers":          "num_workers",
    "output.wandb":                "wandb",
}

_DEFAULTS = {
    "data_dirs": None, "data_dir": "data/mechanism_dataset", "max_samples": None,
    # DiT architecture defaults (DiT-S/4)
    "hidden_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 4,
    "mlp_ratio": 4.0, "bc_channels": 6, "num_design_spaces": 4, "dropout": 0.0,
    "n_concept_layers": 2, "n_concept_heads": 6,
    "bc_token_mode": "full",  # 'full' | 'ds_only' | 'pool_6'
    # UNet keys (unused but kept so shared YAML doesn't break)
    "model_channels": 64, "channel_mult": [1, 2, 3, 4],
    "num_res_blocks": 2, "attention_resolutions": "16,8",
    # flow matching
    "fm_sigma": 0.0, "fm_type": "otcfm", "timestep_sampling": "uniform",
    # physics
    "lambda_physics": 0.1, "physics_start_iter": 5000,
    "physics_ramp_iters": 10000, "physics_min_t": 0.5,
    "physics_every": 1, "physics_batch_size": 4,
    "w_mechanism": 1.0, "w_binary": 1.0,
    "w_compliance": 0.1, "w_tv": 0.05, "penal": 5.0, "rmin": 0.5, "E_min": 1e-4,
    # training
    "batch_size": 8, "lr": 2e-4, "grad_clip": 1.0,
    "total_steps": 300000, "warmup": 5000,
    "ema_decay": 0.9999, "ema_start": 1000,
    "cond_drop_prob": 0.0, "cfg_scale": 1.0,
    "pretrained_encoder": None, "freeze_encoder": False,
    "lambda_cls": 0.1, "encoder_lr_scale": 0.1,
    "lambda_diversity": 0.1,
    "contrastive_margin": 0.01,
    # 2-stage training
    "training_stage": 0,  # 0 = original (all-at-once), 1 = BC-only, 2 = add DS + freeze backbone
    "freeze_backbone": False,
    "disable_ds_tokens": False,
    "stage1_checkpoint": None,
    # DS classifier loss (Stage 2.1)
    "ds_classifier_path": None,
    "lambda_ds_cls": 0.0,
    "ds_cls_min_t": 0.3,
    "ds_cls_start_iter": 0,
    "ds_cls_ramp_iters": 5000,
    # logging
    "log_freq": 20, "eval_freq": 500,
    "sample_freq": 10000, "save_freq": 10000,
    "n_ode_steps": 100, "n_sample_vis": 4,
    # output
    "output_dir": "./trained_models/picfm", "run_name": "run_dit_1",
    "resume": None, "finetune": None, "num_workers": 4, "wandb": False,
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


def build_resolved_yaml_dit(args):
    """Build a self-contained YAML snapshot from resolved args (DiT version)."""
    return {
        "data": {k: getattr(args, k) for k in ["data_dirs", "data_dir", "max_samples"]},
        "model": {k: getattr(args, k, None) for k in [
            "hidden_dim", "depth", "num_heads", "patch_size", "mlp_ratio",
            "bc_channels", "num_design_spaces", "dropout",
        ]},
        "flow_matching": {"sigma": args.fm_sigma, "type": args.fm_type},
        "physics": {k: getattr(args, k) for k in [
            "lambda_physics", "physics_start_iter", "physics_ramp_iters",
            "physics_min_t", "physics_every", "physics_batch_size",
            "w_mechanism", "w_binary", "w_compliance", "w_tv",
            "penal", "rmin", "E_min",
        ]},
        "training": {k: getattr(args, k) for k in [
            "batch_size", "lr", "grad_clip", "total_steps", "warmup",
            "ema_decay", "ema_start", "cond_drop_prob", "cfg_scale",
            "pretrained_encoder", "freeze_encoder", "lambda_cls", "encoder_lr_scale",
            "lambda_diversity",
            "contrastive_margin",
        ]},
        "logging": {k: getattr(args, k) for k in [
            "log_freq", "eval_freq", "sample_freq", "save_freq",
            "n_ode_steps", "n_sample_vis",
        ]},
        "output": {k: getattr(args, k) for k in [
            "output_dir", "run_name", "resume", "finetune", "num_workers", "wandb",
        ]},
    }


def parse_args_dit():
    parser = argparse.ArgumentParser(
        description="PICFM-DiT Training for Mechanism Design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/train_dit.yaml",
                        help="Path to YAML configuration file")

    _CLI_TYPES = {
        "data_dir": str, "max_samples": int,
        # DiT architecture
        "hidden_dim": int, "depth": int, "num_heads": int, "patch_size": int,
        "mlp_ratio": float, "bc_channels": int, "num_design_spaces": int,
        "dropout": float,
        # UNet keys (ignored but parseable)
        "model_channels": int, "num_res_blocks": int,
        "attention_resolutions": str,
        # flow matching
        "fm_sigma": float, "fm_type": str, "timestep_sampling": str,
        # physics
        "lambda_physics": float, "physics_start_iter": int,
        "physics_ramp_iters": int, "physics_min_t": float,
        "physics_every": int, "physics_batch_size": int,
        "w_mechanism": float,
        "w_binary": float, "w_compliance": float, "w_tv": float,
        "penal": float, "rmin": float, "E_min": float,
        # training
        "batch_size": int, "lr": float, "grad_clip": float,
        "total_steps": int, "warmup": int,
        "ema_decay": float, "ema_start": int,
        "cond_drop_prob": float, "cfg_scale": float,
        "pretrained_encoder": str, "lambda_cls": float,
        "encoder_lr_scale": float, "lambda_diversity": float,
        "contrastive_margin": float,
        # DS classifier loss
        "ds_classifier_path": str, "lambda_ds_cls": float,
        "ds_cls_min_t": float,
        # logging
        "log_freq": int, "eval_freq": int,
        "sample_freq": int, "save_freq": int,
        "n_ode_steps": int, "n_sample_vis": int,
        # output
        "output_dir": str, "run_name": str,
        "resume": str, "num_workers": int,
        "finetune": str,
    }
    for name, tp in _CLI_TYPES.items():
        parser.add_argument(f"--{name}", type=tp, default=None)
    parser.add_argument("--data_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--freeze_encoder", action="store_true", default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=None)
    parser.add_argument("--disable_ds_tokens", action="store_true", default=None)
    parser.add_argument("--training_stage", type=int, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)

    cli_args = parser.parse_args()

    cfg_path = Path(cli_args.config)
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        print(f"[Config] Loaded {cfg_path}")
    else:
        yaml_cfg = {}
        print(f"[Config] No YAML at {cfg_path}, using CLI / defaults")

    flat_yaml = _flatten_yaml(yaml_cfg)

    merged = {}
    for flat_key, attr_name in _YAML_KEY_MAP.items():
        cli_val = getattr(cli_args, attr_name, None)
        yaml_val = flat_yaml.get(flat_key)
        if cli_val is not None:
            merged[attr_name] = cli_val
        elif yaml_val is not None:
            merged[attr_name] = yaml_val
        else:
            merged[attr_name] = _DEFAULTS.get(attr_name)

    merged["config"] = str(cfg_path)
    return argparse.Namespace(**merged)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args_dit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.output_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "samples").mkdir(exist_ok=True)

    # Save resolved config
    with open(save_dir / "args.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
    with open(save_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(build_resolved_yaml_dit(args), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    log_fn = lambda d, step: None
    if args.wandb:
        import wandb
        wandb.init(project="picfm_topop", name=args.run_name, config=vars(args))
        log_fn = lambda d, step: wandb.log(d, step=step)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("\n--- Loading Dataset ---")
    if args.data_dirs is not None:
        ds_train = build_multi_dataset(
            args.data_dirs,
            max_samples=args.max_samples,
            return_img=True,
            normalize_geometry=False,
        )
        first_sub = ds_train.datasets[0]
        geom_sample, cond_sample, _ = first_sub[0]
    else:
        ds_train = MechanismConditionedDataset(
            data_dir=args.data_dir,
            return_img=True,
            normalize_geometry=False,
            max_samples=args.max_samples,
        )
        geom_sample, cond_sample, _ = ds_train[0]

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_iter = cycle(dl_train)

    _, H, W = geom_sample.shape
    cond_channels = cond_sample.shape[0]
    print(f"  Geometry: {geom_sample.shape}  Condition: {cond_sample.shape}")
    print(f"  Image size: {H}×{W}  (H×W)  Total samples: {len(ds_train)}")

    # ------------------------------------------------------------------
    # Model — DiT
    # ------------------------------------------------------------------
    print(f"\n--- DiT Model Setup ---")
    print(f"  hidden_dim={args.hidden_dim}, depth={args.depth}, "
          f"num_heads={args.num_heads}, patch_size={args.patch_size}")
    print(f"  mlp_ratio={args.mlp_ratio}, bc_channels={args.bc_channels}")
    bc_token_mode = getattr(args, 'bc_token_mode', 'full')
    print(f"  bc_token_mode={bc_token_mode}")
    print(f"  cond_spatial_channels={cond_channels}")

    model = DiTModel(
        img_height=H,
        img_width=W,
        patch_size=args.patch_size,
        in_channels=1,
        out_channels=1,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        bc_channels=args.bc_channels,
        num_design_spaces=args.num_design_spaces,
        n_concept_layers=getattr(args, 'n_concept_layers', 2),
        n_concept_heads=getattr(args, 'n_concept_heads', 6),
        drop_path_rate=getattr(args, 'drop_path_rate', 0.0),
        cond_spatial_channels=cond_channels,
        bc_token_mode=bc_token_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = (H // args.patch_size) * (W // args.patch_size)
    print(f"  Parameters: {n_params:,}")
    print(f"  Tokens: {n_tokens}  ({H // args.patch_size} × {W // args.patch_size})")

    # ------------------------------------------------------------------
    # Pre-trained encoder (v10: CLIP-like compositional encoder)
    # ------------------------------------------------------------------
    pretrained_enc = getattr(args, 'pretrained_encoder', None)
    freeze_enc = getattr(args, 'freeze_encoder', False)
    lambda_cls = getattr(args, 'lambda_cls', 0.0)
    enc_lr_scale = getattr(args, 'encoder_lr_scale', 0.1)

    ds_classifier = None  # aux classifier head for encoder regularization

    if pretrained_enc:
        print(f"\n--- Loading pre-trained encoder from {pretrained_enc} ---")
        enc_ckpt = torch.load(pretrained_enc, map_location=device, weights_only=False)
        model.cond_encoder.load_state_dict(enc_ckpt["encoder_state_dict"])
        enc_acc = enc_ckpt.get("accuracy", "?")
        enc_epoch = enc_ckpt.get("epoch", "?")
        print(f"  Encoder loaded (epoch={enc_epoch}, accuracy={enc_acc}%)")

        if freeze_enc:
            print(f"  Freezing encoder weights (CLIP-style)")
            for p in model.cond_encoder.parameters():
                p.requires_grad_(False)
        else:
            print(f"  Encoder LR = {args.lr * enc_lr_scale:.1e} "
                  f"({enc_lr_scale}× main LR)")

    # Aux DS classifier on global_vec (prevents encoder collapse during training)
    if lambda_cls > 0 and not freeze_enc:
        ds_classifier = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 4),  # 4 design-space classes
        ).to(device)
        # If we loaded a pre-trained encoder, we could also load the cls head,
        # but a fresh one works fine since global_vec is already discriminative.
        print(f"  Aux DS classifier: lambda_cls={lambda_cls}")
    elif lambda_cls > 0 and freeze_enc:
        print(f"  [Note] Encoder frozen → aux classifier disabled (not needed)")
        lambda_cls = 0.0

    # ------------------------------------------------------------------
    # 2-Stage Training Setup
    # ------------------------------------------------------------------
    training_stage = getattr(args, 'training_stage', 0)
    disable_ds = getattr(args, 'disable_ds_tokens', False)
    freeze_backbone = getattr(args, 'freeze_backbone', False)
    stage1_ckpt_path = getattr(args, 'stage1_checkpoint', None)

    if training_stage == 1:
        # Stage 1: BC-only training (no DS tokens)
        disable_ds = True
        print(f"\n--- STAGE 1: BC-Only Training ---")
        print(f"  DS tokens DISABLED — cross-attn sees only BC spatial tokens")
        print(f"  Model learns topology structure + BC→topology alignment")

    elif training_stage == 2:
        # Stage 2: Add DS tokens, freeze backbone
        disable_ds = False
        freeze_backbone = True
        print(f"\n--- STAGE 2: DS Style Training (backbone frozen) ---")

        # Load Stage 1 checkpoint
        if stage1_ckpt_path:
            print(f"  Loading Stage 1 checkpoint: {stage1_ckpt_path}")
            s1_ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(s1_ckpt["model_state_dict"], strict=False)
            if missing:
                print(f"  [INFO] New v17 params auto-initialized (zero-init): {missing}")
            if unexpected:
                print(f"  [WARNING] Unexpected keys in checkpoint: {unexpected}")
            s1_iter = s1_ckpt.get("iteration", "?")
            print(f"  Stage 1 model loaded (trained {s1_iter} iterations)")
        else:
            print(f"  [WARNING] No stage1_checkpoint specified — using current weights")

        # Freeze backbone, keep only cross-attn + DS params trainable
        n_train, n_froz = model.freeze_backbone_for_stage2()
        print(f"  Backbone frozen: {n_froz:,} params frozen, {n_train:,} trainable")
        print(f"  Trainable: cross-attn layers + DS token bank/proj + concept transformer")

    if disable_ds and training_stage != 1:
        print(f"\n  [INFO] DS tokens disabled via flag (not stage mode)")

    # ------------------------------------------------------------------
    # DS Classifier (Stage 2.1: frozen perceptual judge)
    # ------------------------------------------------------------------
    ds_cls_model = None
    lambda_ds_cls = getattr(args, 'lambda_ds_cls', 0.0)
    ds_cls_min_t = getattr(args, 'ds_cls_min_t', 0.3)
    ds_cls_path = getattr(args, 'ds_classifier_path', None)

    if ds_cls_path and lambda_ds_cls > 0:
        print(f"\n--- DS Classifier (Stage 2.1) ---")
        cls_ckpt = torch.load(ds_cls_path, map_location=device, weights_only=False)
        ds_cls_model = DSClassifierCNN(
            num_classes=cls_ckpt.get('num_classes', 3),
            in_channels=1,
        ).to(device)
        ds_cls_model.load_state_dict(cls_ckpt['model_state_dict'])
        ds_cls_model.eval()
        for p in ds_cls_model.parameters():
            p.requires_grad_(False)
        cls_acc = cls_ckpt.get('accuracy', '?')
        cls_n = cls_ckpt.get('num_classes', '?')
        print(f"  Loaded from: {ds_cls_path}")
        print(f"  Accuracy: {cls_acc}%  Classes: {cls_n}")
        print(f"  lambda_ds_cls={lambda_ds_cls}, min_t={ds_cls_min_t}, start_iter={getattr(args, 'ds_cls_start_iter', 0)}")
        print(f"  Classifier is FROZEN (provides gradient to DiT only)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_params - n_trainable
    if n_frozen > 0:
        print(f"  Trainable: {n_trainable:,}  Frozen: {n_frozen:,}")

    # EMA: exclude frozen params appropriately
    if training_stage == 2:
        # In stage 2, only EMA the trainable params (cross-attn + DS)
        ema = EMA(decay=args.ema_decay, exclude_prefix=None)
    elif freeze_enc:
        ema = EMA(decay=args.ema_decay, exclude_prefix='cond_encoder.')
    else:
        ema = EMA(decay=args.ema_decay, exclude_prefix=None)
    ema.register(model)

    # Optimizer: only trainable params
    if training_stage == 2:
        # Stage 2: simple optimizer on trainable params only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
        print(f"  Stage 2 optimizer: {len(trainable_params)} param groups, lr={args.lr}")
    elif pretrained_enc and not freeze_enc:
        enc_params = list(model.cond_encoder.parameters())
        enc_param_ids = {id(p) for p in enc_params}
        dit_params = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in enc_param_ids]
        param_groups = [
            {"params": dit_params, "lr": args.lr},
            {"params": enc_params, "lr": args.lr * enc_lr_scale},
        ]
        if ds_classifier is not None:
            param_groups.append(
                {"params": list(ds_classifier.parameters()), "lr": args.lr}
            )
        optimizer = optim.AdamW(param_groups, weight_decay=0.0)
    else:
        all_params = list(model.parameters())
        if ds_classifier is not None:
            all_params += list(ds_classifier.parameters())
        optimizer = optim.AdamW(
            [p for p in all_params if p.requires_grad],
            lr=args.lr, weight_decay=0.0,
        )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_lr(step, args.warmup)
    )

    # ------------------------------------------------------------------
    # Flow Matching
    # ------------------------------------------------------------------
    if args.fm_type == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.fm_sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=args.fm_sigma)
    print(f"  Flow Matcher: {args.fm_type} (sigma={args.fm_sigma})")
    print(f"  Timestep sampling: {getattr(args, 'timestep_sampling', 'uniform')}")

    # ------------------------------------------------------------------
    # Physics: base solver
    # ------------------------------------------------------------------
    base_solver = None
    use_physics = args.lambda_physics > 0
    if use_physics:
        print("\n--- Physics Base Solver Setup ---")
        ny_grid, nx_grid = H, W
        dummy_fixed = torch.tensor([0, 1], dtype=torch.long, device=device)
        base_solver = TorchFEMSolverFast(
            nx=nx_grid, ny=ny_grid,
            fixed_dofs=dummy_fixed,
            penal=args.penal,
            rmin=args.rmin,
            E_min=args.E_min,
            device=str(device), dtype=torch.float32,
        )
        print(f"  Grid: {nx_grid}×{ny_grid}  ({nx_grid * ny_grid} elements)")
        print(f"  penal={args.penal}, rmin={args.rmin}, E_min={args.E_min:.1e}")

    # ------------------------------------------------------------------
    # Resume / Fine-tune
    # ------------------------------------------------------------------
    start_iter = 0
    ckpt_path = args.resume or args.finetune
    if ckpt_path:
        is_finetune = args.finetune is not None
        mode = "fine-tuning" if is_finetune else "resuming"
        print(f"\n--- Loading checkpoint ({mode}) from {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  [INFO] New params auto-initialized: {missing}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys in checkpoint: {unexpected}")

        if not is_finetune:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "ema_shadow" in ckpt:
                ema.shadow = ckpt["ema_shadow"]
            start_iter = ckpt.get("iteration", 0)
            print(f"  Resumed at iteration {start_iter}")
        else:
            ema.register(model)
            print(f"  Loaded model weights only (optimizer + scheduler reset)")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    print(f"\nStarting DiT training for {args.total_steps} iterations...")
    print(f"  Physics loss: lambda={args.lambda_physics}, "
          f"start={args.physics_start_iter}, ramp={args.physics_ramp_iters}, "
          f"min_t={args.physics_min_t}")

    pbar = tqdm(range(start_iter, args.total_steps + 1), initial=start_iter,
                total=args.total_steps + 1, dynamic_ncols=True)

    loss_cfm_avg = None
    loss_phys_avg = None
    sub_loss_avg = {"mech": None, "vol": None, "bin": None, "tv": None, "comp": None}
    consecutive_nan_count = 0
    max_consecutive_nan = 50
    last_good_checkpoint = None

    for iteration in pbar:
        model.train()

        # --- Load batch ---
        geom, cond, _ = next(dl_iter)
        geom = geom.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        x_1 = geom
        x_0 = torch.randn_like(x_1)

        # --- Classifier-Free Guidance: per-sample null mask ---
        # No CFG dropout during warmup (let model learn to use conditions first),
        # then ramp to 5% after warmup completes.
        cond_drop_target = getattr(args, 'cond_drop_prob', 0.05)
        if iteration < args.warmup:
            cond_drop_prob = 0.0
        else:
            cond_drop_prob = cond_drop_target
        null_mask = None
        if cond_drop_prob > 0:
            null_mask = torch.rand(cond.shape[0], device=cond.device) < cond_drop_prob

        # --- Flow Matching (with optional lognormal timestep sampling) ---
        if getattr(args, 'timestep_sampling', 'uniform') == 'lognormal':
            # Lumina-T2X / SD3 style: t = sigmoid(N(0,1)), concentrates near t=0.5
            u = torch.randn(x_0.shape[0], device=x_0.device)
            t_pre = torch.sigmoid(u)
        else:
            t_pre = None  # let FM sample uniform internally
        if args.fm_type == 'otcfm':
            # Guided OT-CFM: use index-as-label to track OT permutation
            batch_idx = torch.arange(x_0.shape[0], device=x_0.device)
            t, x_t, u_t, y0_idx, y1_idx = FM.guided_sample_location_and_conditional_flow(
                x_0, x_1, y0=batch_idx, y1=batch_idx, t=t_pre
            )
            # Reorder conditions/noise to match OT-permuted data
            cond = cond[y1_idx.long()]
            x_0 = x_0[y0_idx.long()]
            if null_mask is not None:
                null_mask = null_mask[y1_idx.long()]
        else:
            t, x_t, u_t = FM.sample_location_and_conditional_flow(x_0, x_1, t=t_pre)
        v_t = model(t.to(device), x_t, cond_spatial=cond, null_mask=null_mask,
                    disable_ds=disable_ds)

        # --- CFM loss ---
        loss_cfm = torch.mean((v_t - u_t) ** 2)

        # NaN detection
        if not torch.isfinite(loss_cfm) or v_t.isnan().any():
            consecutive_nan_count += 1
            if consecutive_nan_count == 1 and last_good_checkpoint is not None:
                tqdm.write(f"\n[WARNING] NaN at iter {iteration}! Saving emergency checkpoint...")
                emergency_path = save_dir / f"emergency_pre_nan_{iteration:07d}.pt"
                torch.save(last_good_checkpoint, str(emergency_path))
                tqdm.write(f"  Emergency checkpoint saved: {emergency_path}")
            if consecutive_nan_count <= 5 or consecutive_nan_count % 100 == 0:
                tqdm.write(f"  [NaN iter {iteration}] count={consecutive_nan_count}")
            if consecutive_nan_count >= max_consecutive_nan:
                tqdm.write(f"\n[FATAL] {max_consecutive_nan} consecutive NaN iterations. Halting.")
                break
            continue
        else:
            if consecutive_nan_count > 0:
                tqdm.write(f"  [INFO] Recovered from NaN after {consecutive_nan_count} iterations")
            consecutive_nan_count = 0

        # --- Physics loss (per-sample BCs) ---
        loss_phys = torch.tensor(0.0, device=device)
        sub_losses_batch = []
        compute_physics = (
            use_physics
            and iteration >= args.physics_start_iter
            and iteration % args.physics_every == 0
        )

        if compute_physics:
            try:
                # OT-CFM: v = x₁ - x₀ (constant), so x̂₁ = x₀ + v_θ
                # This is t-independent (same formula as ds_cls block).
                x_1_hat = x_0 + v_t
                rho = x_1_hat[:, 0].clamp(0.0, 1.0).transpose(1, 2)

                t_mask = t >= args.physics_min_t
                if not t_mask.any():
                    loss_phys = torch.tensor(0.0, device=device)
                    compute_physics = False
                else:
                    valid_indices = torch.where(t_mask)[0]
                    t_valid = t[t_mask]
                    t_weight = t_valid.pow(1).mean()
                    sigma_phys = (1.0 - t_valid).clamp(min=0.01).to(device)

                    valid_rho = rho[valid_indices].reshape(len(valid_indices), -1)
                    valid_conds = cond[valid_indices]

                    out_phys = compute_batched_physics_loss(
                        valid_rho, valid_conds, base_solver,
                        sigma=sigma_phys,
                        w_mechanism=args.w_mechanism,
                        w_binary=args.w_binary,
                        w_compliance=args.w_compliance,
                        w_tv=args.w_tv,
                    )

                    if torch.isfinite(out_phys["total"]):
                        loss_phys = out_phys["total"] * t_weight
                        sub_losses_batch = [{
                            "mech": out_phys["mechanism"].item(),
                            "bin":  out_phys["binary"].item(),
                            "tv":   out_phys["tv"].item(),
                            "comp": out_phys["compliance_normalized"].item(),
                        }]
                    else:
                        loss_phys = torch.tensor(0.0, device=device)
                        compute_physics = False

                if not torch.isfinite(loss_phys):
                    if iteration % 500 == 0:
                        tqdm.write(f"  [Warning] Physics loss NaN/Inf at iter {iteration}")
                    loss_phys = torch.tensor(0.0, device=device)
                    compute_physics = False
            except Exception as e:
                if iteration % 500 == 0:
                    tqdm.write(f"  [Warning] Physics exception at iter {iteration}: {e}")
                loss_phys = torch.tensor(0.0, device=device)
                compute_physics = False

        # --- Physics weight ramp ---
        if compute_physics:
            progress = (iteration - args.physics_start_iter) / max(args.physics_ramp_iters, 1)
            phys_weight = args.lambda_physics * min(progress, 1.0)
        else:
            phys_weight = 0.0

        # --- Aux DS classification loss (encoder regularization) ---
        loss_cls = torch.tensor(0.0, device=device)
        if ds_classifier is not None and lambda_cls > 0:
            # Extract ds_label from condition (only for non-null samples)
            ds_label = cond[:, 6:10, 0, 0].argmax(dim=1).long()  # (B,)
            # Get global_vec from encoder (already computed in forward pass,
            # but we need it separately here for the classifier)
            with torch.no_grad() if freeze_enc else torch.enable_grad():
                gv, _, _ = model.cond_encoder(cond, use_null=False)
            cls_logits = ds_classifier(gv)
            # Only compute on non-null samples
            if null_mask is not None:
                valid = ~null_mask
                if valid.any():
                    loss_cls = F.cross_entropy(cls_logits[valid], ds_label[valid])
            else:
                loss_cls = F.cross_entropy(cls_logits, ds_label)

        # --- Total loss ---
        loss_ds_cls = torch.tensor(0.0, device=device)
        ds_cls_start = getattr(args, 'ds_cls_start_iter', 0)
        if ds_cls_model is not None and lambda_ds_cls > 0 and iteration >= ds_cls_start:
            # OT-CFM: v = x₁ - x₀ (constant velocity), so x̂₁ = x₀ + v_θ
            # This is t-independent: prediction error maps 1:1 regardless of t,
            # unlike x_t + (1-t)·v_θ which amplifies error near t=0.
            x1_hat = x_0 + v_t

            # Apply at all timesteps (quality is uniform thanks to x₀ + v_θ)
            t_cls_mask = t >= ds_cls_min_t
            if t_cls_mask.any():
                x1_valid = x1_hat[t_cls_mask]  # (N_valid, 1, H, W)

                # Extract GT ds_label from condition channels 7-10
                # Note: only 3 active classes (topopt=0, finray=1, graph=2)
                # but channels are 7,8,9,10 — the classifier uses 0,1,2
                ds_label_gt = cond[t_cls_mask, 7:10, 0, 0].argmax(dim=1).long()

                # Exclude CFG-null samples if any
                if null_mask is not None:
                    valid_non_null = ~null_mask[t_cls_mask]
                    if valid_non_null.any():
                        x1_valid = x1_valid[valid_non_null]
                        ds_label_gt = ds_label_gt[valid_non_null]
                    else:
                        x1_valid = None

                if x1_valid is not None and x1_valid.shape[0] > 0:
                    # Classifier was trained on [-1,1] range; model output is [0,1]
                    x1_clipped = x1_valid.clamp(0.0, 1.0) * 2.0 - 1.0
                    cls_logits_ds = ds_cls_model(x1_clipped)
                    loss_ds_cls = F.cross_entropy(cls_logits_ds, ds_label_gt)

                    # Sanity: don't backprop NaN from classifier
                    if not torch.isfinite(loss_ds_cls):
                        loss_ds_cls = torch.tensor(0.0, device=device)

        # --- DS classifier weight ramp (mirrors physics ramp) ---
        ds_cls_ramp_iters = getattr(args, 'ds_cls_ramp_iters', 5000)
        if lambda_ds_cls > 0 and iteration >= ds_cls_start and loss_ds_cls.item() > 0:
            ds_cls_progress = (iteration - ds_cls_start) / max(ds_cls_ramp_iters, 1)
            ds_cls_weight = lambda_ds_cls * min(ds_cls_progress, 1.0)
        else:
            ds_cls_weight = 0.0

        loss = (loss_cfm + phys_weight * loss_phys
                + lambda_cls * loss_cls
                + ds_cls_weight * loss_ds_cls)

        # --- Backward ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient NaN check
        grad_nan = False
        grad_norm_pre = 0.0
        for p in model.parameters():
            if p.grad is not None:
                if p.grad.isnan().any() or p.grad.isinf().any():
                    grad_nan = True
                    break
                grad_norm_pre += p.grad.norm().item() ** 2
        grad_norm_pre = grad_norm_pre ** 0.5

        if grad_nan:
            tqdm.write(f"  [WARNING iter {iteration}] Gradient NaN/Inf! Skipping update.")
            tqdm.write(f"    loss_cfm={loss_cfm.item():.3e}, loss_phys={loss_phys.item():.3e}")
            optimizer.zero_grad(set_to_none=True)
            consecutive_nan_count += 1
            if consecutive_nan_count >= max_consecutive_nan:
                tqdm.write(f"\n[FATAL] Too many gradient NaN iterations. Halting.")
                break
            continue

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration > args.ema_start:
            ema.update(model)

        # --- Periodic in-memory checkpoint ---
        if iteration % 1000 == 0:
            last_good_checkpoint = {
                "iteration": iteration,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_shadow": {k: v.cpu().clone() for k, v in ema.shadow.items()} if ema.shadow else None,
            }

        # --- Logging ---
        _cfm_val = loss_cfm.item()
        loss_cfm_avg = _cfm_val if loss_cfm_avg is None else 0.95 * loss_cfm_avg + 0.05 * _cfm_val
        if compute_physics:
            _phys_val = loss_phys.item()
            loss_phys_avg = _phys_val if loss_phys_avg is None else 0.95 * loss_phys_avg + 0.05 * _phys_val
            if sub_losses_batch:
                for key in sub_loss_avg:
                    vals = [s[key] for s in sub_losses_batch]
                    v = sum(vals) / len(vals)
                    sub_loss_avg[key] = v if sub_loss_avg[key] is None else 0.95 * sub_loss_avg[key] + 0.05 * v

        if iteration % args.log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            phys_str = f"phys={loss_phys_avg:.3e}" if loss_phys_avg is not None else "phys=-"
            if sub_loss_avg["mech"] is not None:
                sl = sub_loss_avg
                sub_str = (f" [M={sl['mech']:.2e} V={sl['vol']:.2e} "
                           f"B={sl['bin']:.2e} T={sl['tv']:.2e} C={sl['comp']:.2e}]")
            else:
                sub_str = ""
            cls_str = f" cls={loss_cls.item():.2e}" if lambda_cls > 0 else ""
            ds_cls_str = f" ds_cls={loss_ds_cls.item():.2e}(w={ds_cls_weight:.2e})" if lambda_ds_cls > 0 else ""
            desc = f"cfm={loss_cfm_avg:.3e} {phys_str}{sub_str}{cls_str}{ds_cls_str} w={phys_weight:.3e} lr={lr_now:.1e}"
            pbar.set_description(desc)

            # File-log every 500 iters so div/cs appear in log files
            if iteration % 500 == 0:
                ds_cls_log = f", loss_ds_cls={loss_ds_cls.item():.3e}(w={ds_cls_weight:.2e})" if lambda_ds_cls > 0 else ""
                tqdm.write(f"    loss_cfm={loss_cfm.item():.3e}, loss_phys={loss_phys.item() if compute_physics else 0:.3e}{ds_cls_log}")

            log_dict = {
                "loss_cfm": loss_cfm.item(),
                "loss_cls": loss_cls.item() if lambda_cls > 0 else 0.0,
                "loss_ds_cls": loss_ds_cls.item() if lambda_ds_cls > 0 else 0.0,
                "loss_physics": loss_phys.item() if compute_physics else 0.0,
                "loss_total": loss.item(),
                "physics_weight": phys_weight,
                "lr": lr_now,
                "grad_norm": grad_norm_pre,
            }
            for key, val in sub_loss_avg.items():
                if val is not None:
                    log_dict[f"sub/{key}"] = val
            log_fn(log_dict, step=iteration)

            if grad_norm_pre > 100:
                tqdm.write(f"  [WARNING iter {iteration}] Large gradient norm: {grad_norm_pre:.1f}")

        # --- Validation ---
        if iteration % args.eval_freq == 0 and iteration > 0:
            model.eval()
            ema.apply_shadow(model)
            with torch.no_grad():
                geom_v, cond_v, _ = next(dl_iter)
                geom_v = geom_v.to(device)
                cond_v = cond_v.to(device)
                x_0_v = torch.randn_like(geom_v)
                if args.fm_type == 'otcfm':
                    batch_idx_v = torch.arange(x_0_v.shape[0], device=x_0_v.device)
                    t_v, x_t_v, u_t_v, _, y1_idx_v = FM.guided_sample_location_and_conditional_flow(
                        x_0_v, geom_v, y1=batch_idx_v
                    )
                    cond_v = cond_v[y1_idx_v.long()]
                else:
                    t_v, x_t_v, u_t_v = FM.sample_location_and_conditional_flow(x_0_v, geom_v)
                v_t_v = model(t_v.to(device), x_t_v, cond_spatial=cond_v,
                              disable_ds=disable_ds)
                loss_val = torch.mean((v_t_v - u_t_v) ** 2).item()
            tqdm.write(f"  [Iter {iteration}] val_cfm={loss_val:.3e}")
            log_fn({"loss_val_cfm": loss_val}, step=iteration)
            ema.restore(model)

        # --- Sample & Visualize ---
        if iteration % args.sample_freq == 0 and iteration > 0:
            tqdm.write(f"  Generating samples at iteration {iteration}...")
            model.eval()
            ema.apply_shadow(model)

            geom_s, cond_s, _ = next(dl_iter)
            n_vis = min(args.n_sample_vis, geom_s.shape[0])
            geom_s = geom_s[:n_vis].to(device)
            cond_s = cond_s[:n_vis].to(device)

            with torch.no_grad():
                samples = euler_sample_dit(
                    model, cond_s,
                    shape=(n_vis, 1, H, W),
                    n_steps=args.n_ode_steps,
                    device=str(device),
                    cfg_scale=getattr(args, 'cfg_scale', 1.0),
                    disable_ds=disable_ds,
                )

            vis_path = save_dir / "samples" / f"iter_{iteration:07d}.png"
            visualize_samples_dit(samples, geom_s, cond_s, str(vis_path), n_show=n_vis)
            tqdm.write(f"  Saved samples to {vis_path}")

            # Diagnostic: same BCs, different design-space labels
            # (skip in Stage 1 — no DS tokens, comparison is meaningless)
            if not disable_ds:
                cmp_path = save_dir / "samples" / f"label_cmp_{iteration:07d}.png"
                visualize_label_comparison_dit(
                    model, cond_s[0].cpu(),
                    str(cmp_path),
                    n_ode_steps=args.n_ode_steps,
                    cfg_scale=getattr(args, 'cfg_scale', 1.0),
                    device=str(device),
                )
                tqdm.write(f"  Saved label comparison to {cmp_path}")

            if use_physics:
                try:
                    rho_gen = samples[:, 0].clamp(0, 1).transpose(1, 2).reshape(n_vis, -1)
                    engine_s = setup_physics_from_condition(
                        cond_s[0], base_solver,
                        device=str(device),
                        w_mechanism=args.w_mechanism,
                        w_binary=args.w_binary,
                        w_compliance=args.w_compliance,
                        w_tv=args.w_tv,
                    )
                    metrics = engine_s.compute_metrics(rho_gen[0])
                    tqdm.write(
                        f"  Physics metrics: "
                        f"bin={metrics['binary_score']:.3f} "
                        f"R_out={metrics['reaction_force']:.4f}"
                    )
                    log_fn({
                        "sample_binary_score": metrics["binary_score"],
                        "sample_reaction_force": metrics["reaction_force"],
                    }, step=iteration)
                except Exception as e:
                    tqdm.write(f"  [Warning] Sampling physics metrics failed: {e}")

            ema.restore(model)

        # --- Checkpoint ---
        if iteration % args.save_freq == 0 and iteration > 0:
            ckpt_save = save_dir / f"checkpoint_{iteration:07d}.pt"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_shadow": ema.shadow,
                "args": vars(args),
                "architecture": "DiT",
                "training_stage": training_stage,
            }, str(ckpt_save))
            tqdm.write(f"  Saved checkpoint: {ckpt_save}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = save_dir / "model_final.pt"
    torch.save({
        "iteration": args.total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "args": vars(args),
        "architecture": "DiT",
        "training_stage": training_stage,
    }, str(final_path))
    print(f"\nDiT training complete. Final model saved to {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
