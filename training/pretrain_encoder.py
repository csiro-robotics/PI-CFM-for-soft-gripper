"""
Pre-train SpatialStyleEncoder (v2) for DiT.

Trains each stream independently with explicit supervision so the encoder
learns meaningful representations before being plugged into a DiT.

Architecture:
  BC stream:  CNN(channels 0-5) → 512 spatial tokens   (bc_cnn + bc_pos_embed)
  DS stream:  ds_class_embeddings blended via one-hot → ds_refine → 512 tokens
  Fusion:     bc + ds → fusion_norm + fusion_proj → context tokens

Training paths (decoupled):
  Path A (reconstruction): bc_cnn → bc_tokens → decoder → MSE on BC channels
           Trains: bc_cnn, bc_pos_embed
  Path B (classification): ds_class_embeddings → blend → ds_refine →
           mean_pool → classifier → CE on ds_label
           Trains: ds_class_embeddings, ds_refine, ds_pos_embed
  Path C (fusion):         bc + ds → fusion_proj → mean_pool → classifier
           Trains: fusion_norm, fusion_proj
           (Turned on after Path A+B converge, epoch > fusion_start)

Concept-swap augmentation randomises DS labels while keeping BCs fixed,
so the two streams learn independent features.

Usage:
  python pretrain_encoder_v2.py
  python pretrain_encoder_v2.py --epochs 150 --lr 1e-3
  python pretrain_encoder_v2.py --data_dirs data/mechanism_dataset:topopt \
      data/finray_dataset:finray data/graph_dataset:graph
"""

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flows.design_encoder import SpatialStyleEncoder, ConditionDecoder
from src.mechanism_dataset import build_multi_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train SpatialStyleEncoder v2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_ds", type=int, default=4)

    # Loss weights
    parser.add_argument("--w_recon", type=float, default=1.0,
                        help="Weight for BC reconstruction loss")
    parser.add_argument("--w_cls_ds", type=float, default=1.0,
                        help="Weight for DS stream classification loss")
    parser.add_argument("--w_cls_fused", type=float, default=1.0,
                        help="Weight for fused context classification loss")
    parser.add_argument("--w_contra", type=float, default=1.0,
                        help="Weight for contrastive loss on ds embeddings")

    # Augmentation
    parser.add_argument("--swap_prob", type=float, default=0.3,
                        help="Probability of concept-swap (swap DS between samples)")
    parser.add_argument("--null_prob", type=float, default=0.15,
                        help="Probability of null token augmentation")

    # Fusion schedule
    parser.add_argument("--fusion_start", type=int, default=30,
                        help="Epoch to start training fusion path (Path C)")

    # Data
    parser.add_argument("--data_dirs", type=str, nargs="+",
                        default=["data/mechanism_dataset:topopt",
                                 "data/finray_dataset:finray",
                                 "data/graph_dataset:graph"])
    parser.add_argument("--output_dir", type=str,
                        default="trained_models/design_encoder")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    print("\n--- Loading Dataset ---")
    ds = build_multi_dataset(
        args.data_dirs,
        max_samples=None, return_img=True, normalize_geometry=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    print(f"  {len(ds)} samples, batch_size={args.batch_size}")

    # ---- Build encoder + auxiliary heads ----
    D = args.hidden_dim
    encoder = SpatialStyleEncoder(
        hidden_dim=D, num_ds=args.num_ds,
    ).to(device)

    # Decoder: reconstructs BC channels from raw bc_tokens
    decoder = ConditionDecoder(out_channels=10, hidden_dim=D).to(device)

    # DS classifier: operates on mean-pooled DS tokens (Path B)
    ds_cls_head = nn.Sequential(
        nn.Linear(D, D // 2),
        nn.GELU(),
        nn.Linear(D // 2, args.num_ds),
    ).to(device)

    # Fused classifier: operates on mean-pooled fused context (Path C)
    fused_cls_head = nn.Sequential(
        nn.Linear(D, D // 2),
        nn.GELU(),
        nn.Linear(D // 2, args.num_ds),
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    n_cls = sum(p.numel() for p in ds_cls_head.parameters())
    print(f"\n  Encoder: {n_enc:,} params")
    print(f"  Decoder: {n_dec:,} params (not saved)")
    print(f"  DS cls head: {n_cls:,} params (not saved)")
    print(f"  Fused cls head: {n_cls:,} params (not saved)")

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(ds_cls_head.parameters()) +
        list(fused_cls_head.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Channel weights for reconstruction: BCs=1, DS=0
    ch_weights = torch.tensor(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        device=device, dtype=torch.float32,
    ).view(1, 10, 1, 1)

    # ---- Training ----
    print(f"\nTraining plan:")
    print(f"  Epochs 1-{args.fusion_start}: Path A (BC recon) + Path B (DS cls) decoupled")
    print(f"  Epochs {args.fusion_start+1}-{args.epochs}: + Path C (fused context cls)")
    print(f"\n{'Epoch':>6} {'Loss':>8} {'Recon':>8} {'DS_CE':>8} "
          f"{'DS_Acc':>7} {'Fus_Acc':>7} {'LR':>10} {'Time':>6}")
    print("-" * 70)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()
        ds_cls_head.train()
        fused_cls_head.train()

        use_fusion = epoch > args.fusion_start

        total_loss = 0
        total_recon = 0
        total_ds_ce = 0
        total_fused_ce = 0
        ds_correct = 0
        fused_correct = 0
        total_samples = 0
        n_batch = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{args.epochs}",
                    leave=False, dynamic_ncols=True)
        for geom, cond, _ in pbar:
            cond = cond.to(device)
            B_cur = cond.shape[0]

            # DS labels from one-hot
            ds_label = cond[:, 6:10, 0, 0].argmax(dim=1).long()

            # ---- Concept Swap Augmentation ----
            if torch.rand(1).item() < args.swap_prob:
                half = B_cur // 2
                idx_swap = torch.cat([torch.arange(half, B_cur),
                                      torch.arange(0, half)]).to(device)
                ds_label = ds_label[idx_swap]
                cond = cond.clone()
                cond[:, 6:10] = cond[idx_swap, 6:10]

            # ---- Null augmentation ----
            use_null = torch.rand(1).item() < args.null_prob

            if use_null:
                # Null path: just train null tokens lightly
                bc_tokens = encoder.null_context.expand(B_cur, -1, -1)
                recon = decoder(bc_tokens)
                loss_recon = (ch_weights * (recon - cond).pow(2)).mean()
                loss = args.w_recon * loss_recon * 0.1  # light null regularization
                loss_ds_ce = torch.tensor(0.0, device=device)
                loss_fused_ce = torch.tensor(0.0, device=device)
            else:
                bc = cond[:, :encoder.bc_channels]
                ds_maps = cond[:, encoder.ds_channel_offset:
                               encoder.ds_channel_offset + encoder.num_ds]

                # ---- Path A: BC reconstruction (bc_cnn only) ----
                bc_tokens = encoder.encode_bc(bc)               # (B, 512, D)
                recon = decoder(bc_tokens)
                loss_recon = (ch_weights * (recon - cond).pow(2)).mean()

                # ---- Path B: DS classification (ds stream only) ----
                ds_tokens = encoder.encode_ds_spatial(ds_maps)  # (B, 512, D)
                ds_pooled = ds_tokens.mean(dim=1)               # (B, D)
                ds_logits = ds_cls_head(ds_pooled)
                loss_ds_ce = F.cross_entropy(ds_logits, ds_label)

                # Contrastive: push different-DS embeddings apart
                loss_contra = torch.tensor(0.0, device=device)
                if args.w_contra > 0:
                    emb_norm = F.normalize(ds_pooled, dim=1)
                    sim = emb_norm @ emb_norm.T
                    same = ds_label.unsqueeze(0) == ds_label.unsqueeze(1)
                    diff = ~same
                    if diff.any():
                        loss_contra = F.relu(sim[diff]).mean()
                    if same.sum() > B_cur:  # more than diagonal
                        loss_contra = loss_contra + (1 - sim[same]).mean() * 0.5

                # ---- Path C: Fused classification (full pipeline) ----
                loss_fused_ce = torch.tensor(0.0, device=device)
                if use_fusion:
                    fused = bc_tokens + ds_tokens
                    context = encoder.fusion_proj(encoder.fusion_norm(fused))
                    fused_pooled = context.mean(dim=1)
                    fused_logits = fused_cls_head(fused_pooled)
                    loss_fused_ce = F.cross_entropy(fused_logits, ds_label)

                loss = (args.w_recon * loss_recon +
                        args.w_cls_ds * loss_ds_ce +
                        args.w_contra * loss_contra)
                if use_fusion:
                    loss = loss + args.w_cls_fused * loss_fused_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_ds_ce += loss_ds_ce.item()
            total_fused_ce += loss_fused_ce.item()
            if not use_null:
                ds_correct += (ds_logits.argmax(1) == ds_label).sum().item()
                if use_fusion:
                    fused_correct += (fused_logits.argmax(1) == ds_label).sum().item()
                total_samples += B_cur
            n_batch += 1

            ds_acc_run = ds_correct / max(total_samples, 1) * 100
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             ds_acc=f"{ds_acc_run:.1f}%")

        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / n_batch
        avg_recon = total_recon / n_batch
        avg_ds_ce = total_ds_ce / n_batch
        ds_acc = ds_correct / max(total_samples, 1) * 100
        fused_acc = fused_correct / max(total_samples, 1) * 100 if use_fusion else 0
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch <= 10 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{epoch:>6} {avg_loss:>8.4f} {avg_recon:>8.4f} {avg_ds_ce:>8.4f} "
                  f"{ds_acc:>6.1f}% {fused_acc:>6.1f}% {lr_now:>10.2e} {dt:>5.1f}s")

        # Save best (based on DS accuracy)
        if ds_acc > best_acc and ds_acc > 90:
            best_acc = ds_acc
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "epoch": epoch,
                "ds_accuracy": ds_acc,
                "args": vars(args),
            }, save_dir / "encoder_best.pt")

    # ---- Save final ----
    final_path = save_dir / "encoder_final.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "epoch": args.epochs,
        "ds_accuracy": ds_acc,
        "args": vars(args),
    }, final_path)
    print(f"\nSaved encoder: {final_path}")

    # ======================================================================
    # Post-training diagnostic
    # ======================================================================
    print("\n" + "=" * 60)
    print("POST-TRAINING DIAGNOSTIC")
    print("=" * 60)

    encoder.eval()
    ds_cls_head.eval()
    fused_cls_head.eval()

    test_loader = DataLoader(ds, batch_size=64, shuffle=True)
    geom, cond, _ = next(iter(test_loader))
    cond = cond.to(device)
    ds_label = cond[:, 6:10, 0, 0].argmax(dim=1).long()
    DS_NAMES = ['topopt', 'finray', 'graph', 'lattice']

    with torch.no_grad():
        context = encoder(cond)

    # 1. DS class embedding separation
    print("\n  DS class embedding cosine similarities:")
    emb = encoder.ds_class_embeddings.data  # (K, D)
    emb_norm = F.normalize(emb, dim=1)
    cos_mat = emb_norm @ emb_norm.T
    for i in range(min(args.num_ds, len(DS_NAMES))):
        for j in range(i + 1, min(args.num_ds, len(DS_NAMES))):
            print(f"    cos({DS_NAMES[i]}, {DS_NAMES[j]}) = {cos_mat[i, j]:.4f}")

    # 2. DS stream classification accuracy
    ds_maps = cond[:, encoder.ds_channel_offset:
                    encoder.ds_channel_offset + encoder.num_ds]
    with torch.no_grad():
        ds_tokens = encoder.encode_ds_spatial(ds_maps)
        ds_logits = ds_cls_head(ds_tokens.mean(dim=1))
    ds_acc_final = (ds_logits.argmax(1) == ds_label).float().mean().item() * 100
    print(f"\n  DS stream accuracy: {ds_acc_final:.1f}%")

    # 3. Fused context classification
    with torch.no_grad():
        fused_logits = fused_cls_head(context.mean(dim=1))
    fused_acc_final = (fused_logits.argmax(1) == ds_label).float().mean().item() * 100
    print(f"  Fused context accuracy: {fused_acc_final:.1f}%")

    # 4. Same BCs, swap DS — does context change?
    bc_sample = cond[0:1]
    cond_a = bc_sample.clone()
    cond_b = bc_sample.clone()
    cond_a[:, 6:10] = 0; cond_a[:, 6] = 1  # topopt
    cond_b[:, 6:10] = 0; cond_b[:, 7] = 1  # finray
    with torch.no_grad():
        ctx_a = encoder(cond_a)
        ctx_b = encoder(cond_b)
    cos_swap = F.cosine_similarity(
        ctx_a.mean(dim=1), ctx_b.mean(dim=1)
    ).item()
    print(f"  Same BCs, swap DS → context cos = {cos_swap:.4f}")

    # 5. CFG contrast
    with torch.no_grad():
        ctx_real = encoder(cond[:8], use_null=False)
        ctx_null = encoder(cond[:8], use_null=True)
    cos_cfg = F.cosine_similarity(
        ctx_real.mean(dim=1), ctx_null.mean(dim=1), dim=1
    ).mean().item()
    print(f"  CFG cos(real, null) = {cos_cfg:.4f}")

    # 6. Reconstruction (BC channels)
    decoder.eval()
    with torch.no_grad():
        bc_tokens = encoder.encode_bc(cond[:16, :encoder.bc_channels])
        recon = decoder(bc_tokens)
    for ch, name in [(0, 'BC_in_x'), (1, 'BC_in_y'), (4, 'BC_out_x'), (5, 'BC_out_y')]:
        gt_flat = cond[:16, ch].flatten()
        re_flat = recon[:, ch].flatten()
        if gt_flat.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([re_flat, gt_flat]))[0, 1].item()
            print(f"  Recon corr ch{ch} ({name}): {corr:.4f}")

    passed = cos_swap < 0.7 and ds_acc_final > 95
    print(f"\n  VERDICT: {'PASS' if passed else 'NEEDS MORE TRAINING'}")
    print(f"  Encoder saved to: {save_dir}/")
    print(f"  Use in DiT with: --pretrained_encoder {final_path}")


if __name__ == "__main__":
    main()
