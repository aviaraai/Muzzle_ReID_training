"""
scripts/train_dinov2_arcface.py — Fine-tune DINOv2 + Sub-center ArcFace

300-Cattle muzzle dataset.
  Train = gallery (3 images/cattle → 900 images)
  Val   = query   (~7 images/cattle → ~2000 images)

Freeze Schedule:
  Epoch  1–10 : Blocks 10–11 + LayerNorm + Head
  Epoch 11–30 : Blocks 8–11  + LayerNorm + Head
  Epoch 31–50 : Blocks 6–11  + LayerNorm + Head

Saves:
  checkpoints/best_top1_encoder.pt
  checkpoints/best_top1_arcface.pt
  checkpoints/best_gap_encoder.pt
  checkpoints/best_gap_arcface.pt
  checkpoints/last_encoder.pt
  checkpoints/last_arcface.pt
  results/metrics.csv

Usage:
  uv run python scripts/train_dinov2_arcface.py --batch-size 8 --grad-accum 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from losses import (
    arcface_parameter_groups,
    build_arcface,
    load_arcface,
    save_arcface,
)
from model import GodhaarModel, build_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "CattleMuzzle"
# benchmark_manifest.csv lives in benchmark/ on the main machine, data/ in the for_aditya package
_manifest_candidates = [
    ROOT / "benchmark" / "benchmark_manifest.csv",
    ROOT / "data" / "benchmark_manifest.csv",
]
MANIFEST = next(
    (p for p in _manifest_candidates if p.exists()), _manifest_candidates[1]
)
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# =============================================================================
# SETTINGS
# =============================================================================

IMG_SIZE = 518  # DINOv2 vit_base_patch14 requires 518×518
EMB_DIM = 768
PROJ_DIM = 256
SUBCENTER_K = 3
ARCFACE_SCALE = 64
ARCFACE_MARGIN = 0.5  # ~28.6°

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("godhaar.train")


# =============================================================================
# DATASET
# =============================================================================


class MuzzleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, label_map: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cattle_id = row["CattleID"]
        img_name = row["Image"]
        img_path = DATA_ROOT / cattle_id / img_name
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.label_map[cattle_id]
        return img, label


# =============================================================================
# EVALUATION — fully vectorised (matmul, no Python loops)
# =============================================================================


@torch.no_grad()
def evaluate(model, loss_fn, gallery_loader, query_loader, device):
    model.eval()

    gallery_embs, gallery_labels = [], []
    for imgs, lbls in gallery_loader:
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            e = model(imgs.to(device))
        gallery_embs.append(e.cpu().numpy())
        gallery_labels.extend(lbls.tolist())
    gallery_embs = np.vstack(gallery_embs).astype(np.float32)
    gallery_labels = np.array(gallery_labels)

    query_embs, query_labels = [], []
    vl = 0.0
    for imgs, lbls in query_loader:
        imgs_dev = imgs.to(device)
        lbls_dev = lbls.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            e = model(imgs_dev)
            loss = loss_fn(e.float(), lbls_dev)
        vl += loss.item()
        query_embs.append(e.cpu().numpy())
        query_labels.extend(lbls.tolist())
    query_embs = np.vstack(query_embs).astype(np.float32)
    query_labels = np.array(query_labels)

    # Whether this run's gallery(train)/query(val) split shares any identity.
    # The ORIGINAL foreign-cattle setup this function was written for does:
    # gallery = 3 images/cattle, query = ~7 MORE images of the SAME cattle —
    # a closed-set, image-disjoint split, so cross-comparing query embeddings
    # against the gallery index finds real genuine pairs.
    #
    # An identity-DISJOINT split (val identities entirely absent from
    # training — the open-set generalization test this project actually
    # wants) breaks that assumption completely: no query image can ever have
    # a genuine match in the gallery, because its identity was never put
    # there. Confirmed by running this once, unmodified, against exactly
    # that kind of split: top1=0.0000, gap=nan (sim[same] was an empty
    # slice) — and since `best_top1.pt` is only saved when
    # metrics["top1"] > best_top1 (starts at 0.0), a run like that would
    # silently produce NO best checkpoint at all, start to finish.
    #
    # Detect which case this is and evaluate accordingly, rather than assume.
    gallery_id_set = set(gallery_labels.tolist())
    query_id_set = set(query_labels.tolist())
    identity_disjoint = gallery_id_set.isdisjoint(query_id_set)

    if not identity_disjoint:
        # Original closed-set protocol, unchanged.
        index = faiss.IndexFlatIP(PROJ_DIM)
        index.add(np.ascontiguousarray(gallery_embs))
        _, ids = index.search(np.ascontiguousarray(query_embs), k=5)

        retrieved_labels = gallery_labels[ids]  # (Q, 5)
        q_col = query_labels[:, None]  # (Q, 1)
        top1 = float((retrieved_labels[:, 0] == query_labels).mean())
        top5 = float((retrieved_labels == q_col).any(axis=1).mean())

        sim = query_embs @ gallery_embs.T  # (Q, G)
        same = query_labels[:, None] == gallery_labels[None, :]

        gen_mean = float(sim[same].mean())
        imp_scores = sim[~same]
        rng = np.random.default_rng(42)
        if len(imp_scores) > 50_000:
            imp_scores = rng.choice(imp_scores, 50_000, replace=False)
        imp_mean = float(imp_scores.mean())
        gap = gen_mean - imp_mean
    else:
        # Open-set generalization test: leave-one-out retrieval WITHIN the
        # held-out query set itself (same protocol as this project's own
        # scratch_ablation_pca.py / scratch_rerank_eval.py) — per-query-image
        # score against every OTHER query image, per-identity aggregated by
        # MAX (matching go-apiserver's own cattleScores aggregation), ranked.
        # This is the number that actually answers "does the fine-tune
        # generalize to cattle it never saw," which is what this split was
        # built for.
        sim_qq = query_embs @ query_embs.T  # (Q, Q)
        n = len(query_labels)
        ranks, gen_scores, imp_scores = [], [], []
        for i in range(n):
            per_id_max: dict[int, float] = {}
            for j in range(n):
                if j == i:
                    continue
                s = float(sim_qq[i, j])
                lbl = int(query_labels[j])
                if lbl not in per_id_max or s > per_id_max[lbl]:
                    per_id_max[lbl] = s
            own_id = int(query_labels[i])
            if own_id not in per_id_max:
                continue  # singleton identity in the query set (no other photo to compare)
            order = sorted(per_id_max.items(), key=lambda kv: -kv[1])
            rank = [ident for ident, _ in order].index(own_id) + 1
            ranks.append(rank)
            gen_scores.append(per_id_max[own_id])
            imp_scores.append(max(s for ident, s in per_id_max.items() if ident != own_id)
                               if len(per_id_max) > 1 else float("nan"))

        ranks_arr = np.array(ranks)
        top1 = float((ranks_arr <= 1).mean()) if len(ranks_arr) else 0.0
        top5 = float((ranks_arr <= 5).mean()) if len(ranks_arr) else 0.0
        gen_mean = float(np.mean(gen_scores)) if gen_scores else float("nan")
        valid_imp = [s for s in imp_scores if not np.isnan(s)]
        imp_mean = float(np.mean(valid_imp)) if valid_imp else float("nan")
        gap = gen_mean - imp_mean if valid_imp else float("nan")

    # val_loss still comes from the ArcFace loss on query images — informational
    # only when identity_disjoint (those classes' ArcFace weight rows never
    # receive a gradient, since their images never appear in the train loop),
    # not used for checkpoint selection either way (top1/gap drive that).
    val_loss = vl / max(len(query_loader), 1)

    metrics = {
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "genuine_mean": round(gen_mean, 4),
        "impostor_mean": round(imp_mean, 4),
        "gap": round(gap, 4),
    }

    return metrics, val_loss, gallery_embs, gallery_labels


# =============================================================================
# HARD NEGATIVE MINING
# =============================================================================


def compute_hard_negatives(gallery_embs, gallery_labels, top_k=20):
    index = faiss.IndexFlatIP(PROJ_DIM)
    index.add(gallery_embs)
    _, neigh = index.search(gallery_embs, top_k + 1)

    hard_dict: dict[int, list[int]] = defaultdict(list)
    for i, nbrs in enumerate(neigh):
        for nb in nbrs[1:]:
            if gallery_labels[nb] != gallery_labels[i]:
                hard_dict[i].append(int(nb))
        hard_dict[i] = hard_dict[i][:top_k]
    return hard_dict


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--gpu-preset",
        choices=["rtx3050", "rtx4060", "rtx4090", "auto"],
        default="auto",
        help="GPU preset: auto-tunes batch-size, grad-accum and precision",
    )
    args = parser.parse_args()

    # ── GPU preset overrides ──────────────────────────────────────────────────
    AMP_DTYPE = "float16"  # default (safe for all GPUs)
    USE_COMPILE = False

    if args.gpu_preset == "auto":
        # detect Ada Lovelace (sm_89) or Ampere (sm_80+) automatically
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                args.gpu_preset = "rtx4060"  # treat Ampere/Ada as high-end
            else:
                args.gpu_preset = "rtx3050"

    if args.gpu_preset in ("rtx4060", "rtx4090"):
        # Ada Lovelace / high-end Ampere: native BF16 + compile
        if args.batch_size == 8 and args.grad_accum == 4:
            # only override defaults if user didn't manually set them
            args.batch_size = 16
            args.grad_accum = 2
        AMP_DTYPE = "bfloat16"
        USE_COMPILE = torch.cuda.is_available() and platform.system() != "Windows"
        log.info(
            "[GPU Preset: %s] batch=%d  accum=%d  amp=bfloat16  compile=%s",
            args.gpu_preset,
            args.batch_size,
            args.grad_accum,
            USE_COMPILE,
        )
    else:
        log.info(
            "[GPU Preset: rtx3050] batch=%d  accum=%d  amp=float16  compile=False",
            args.batch_size,
            args.grad_accum,
        )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # File Logging
    fh = logging.FileHandler(
        RESULTS_DIR / "training.log", mode="a" if args.resume else "w", encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
        torch.backends.cudnn.benchmark = True

    log.info("=" * 65)
    log.info("Godhaar — DINOv2 + Sub-center ArcFace Fine-tuning")
    log.info(f"  Epochs     : {args.epochs}")
    log.info(
        f"  Batch      : {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}"
    )
    log.info(f"  LR         : backbone={args.lr_backbone}  head={args.lr_head}")
    log.info("=" * 65)

    # ── Manifest ─────────────────────────────────────────────────────────────
    df = pd.read_csv(MANIFEST, dtype=str)
    train_df = df[df["Split"] == "gallery"].reset_index(drop=True)
    val_df = df[df["Split"] == "query"].reset_index(drop=True)

    all_ids = sorted(df["CattleID"].unique().tolist())
    label_map = {cid: i for i, cid in enumerate(all_ids)}
    num_classes = len(label_map)

    log.info(f"  Classes    : {num_classes}")
    log.info(f"  Train      : {len(train_df):,}  (gallery)")
    log.info(f"  Val        : {len(val_df):,}  (query)")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.07
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = MuzzleDataset(train_df, train_transform, label_map)
    gallery_dataset = MuzzleDataset(train_df, val_transform, label_map)
    val_dataset = MuzzleDataset(val_df, val_transform, label_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    query_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    if args.resume:
        log.info("Resuming from checkpoint...")
        model, ckpt = GodhaarModel.load_checkpoint(CKPT_DIR / "last.pt", device=device)
        loss_fn = build_arcface(num_classes=num_classes, device=device)
        if "arcface_state" in ckpt:
            loss_fn.load_state_dict(ckpt["arcface_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        ckpt_encoder = ckpt  # for loading optimizer states
    else:
        log.info("Loading DINOv2-base...")
        model = build_model(
            num_classes=num_classes,
            device=device,
            pooling="gem",
        )
        loss_fn = build_arcface(
            num_classes=num_classes,
            device=device,
        )
        start_epoch = 1

    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
        )
        + arcface_parameter_groups(
            loss_fn,
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        ),
    )

    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    # BF16 doesn't need a loss scaler (no underflow risk); FP16 does.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and AMP_DTYPE == "float16")
    )

    # torch.compile: Ada Lovelace (RTX 40xx) benefits most
    if USE_COMPILE:
        try:
            log.info("Compiling model with torch.compile (mode=reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            log.warning("torch.compile unavailable: %s", e)
            log.warning("Continuing without torch.compile.")

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not args.resume:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "top1",
                    "top5",
                    "genuine_mean",
                    "impostor_mean",
                    "gap",
                    "lr_backbone",
                    "lr_head",
                ]
            )
    else:
        optimizer.load_state_dict(ckpt_encoder["optimizer_state"])
        scheduler.load_state_dict(ckpt_encoder["scheduler_state"])

    if args.resume:
        best_top1 = ckpt_encoder.get("metrics", {}).get("top1", 0.0)
        best_gap = ckpt_encoder.get("metrics", {}).get("gap", 0.0)
    else:
        best_top1 = 0.0
        best_gap = 0.0
    hard_negs: dict[int, list[int]] = {}
    cached_gallery_embs = None
    cached_gallery_labels = None

    patience = 15
    epochs_no_improve = 0
    best_epoch = start_epoch - 1

    log.info("\nStarting training...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.progressive_unfreeze(epoch)

        # Hard negative mining (every 5 epochs)
        if epoch % 5 == 0 and cached_gallery_embs is not None:
            hard_negs = compute_hard_negatives(
                cached_gallery_embs, cached_gallery_labels, top_k=20
            )
            log.info(f"  Hard negatives: {len(hard_negs)} anchors")

        # ── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (imgs, labels) in enumerate(train_loader):
            # Inject hard negatives
            if hard_negs:
                start = step * args.batch_size
                extra_imgs, extra_labels = [], []
                for idx in range(
                    start, min(start + args.batch_size, len(train_dataset))
                ):
                    pool = hard_negs.get(idx, [])
                    if pool:
                        ni = random.choice(pool)
                        neg_img, neg_label = gallery_dataset[ni]
                        extra_imgs.append(neg_img)
                        extra_labels.append(neg_label)
                if extra_imgs:
                    imgs = torch.cat([imgs, torch.stack(extra_imgs)])
                    labels = torch.cat(
                        [labels, torch.tensor(extra_labels, dtype=torch.long)]
                    )

            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=getattr(torch, AMP_DTYPE),
                enabled=(device.type == "cuda"),
            ):
                embeddings = model(imgs)
                loss = (
                    loss_fn(
                        embeddings.float(),
                        labels,
                    )
                    / args.grad_accum
                )
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

            running_loss += loss.item() * args.grad_accum
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # ── RETRIEVAL EVAL & VAL LOSS ────────────────────────────────────────
        t_eval = time.time()
        metrics, val_loss, cached_gallery_embs, cached_gallery_labels = evaluate(
            model, loss_fn, gallery_loader, query_loader, device
        )
        eval_s = time.time() - t_eval

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[4]["lr"]

        log.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"TrLoss={train_loss:.4f} VlLoss={val_loss:.4f} | "
            f"Top1={metrics['top1']:.4f} Top5={metrics['top5']:.4f} Gap={metrics['gap']:.4f} | "
            f"LR={lr_bb:.1e}/{lr_head:.1e} | "
            f"{time.time() - t0:.0f}s (eval {eval_s:.1f}s)"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    round(train_loss, 4),
                    round(val_loss, 4),
                    metrics["top1"],
                    metrics["top5"],
                    metrics["genuine_mean"],
                    metrics["impostor_mean"],
                    metrics["gap"],
                    f"{lr_bb:.2e}",
                    f"{lr_head:.2e}",
                ]
            )

        # ── CHECKPOINTS ──────────────────────────────────────────────────────
        shared_ckpt_kwargs = dict(
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics=metrics,
            train_loss=train_loss,
            val_loss=val_loss,
            arcface_state=loss_fn.state_dict(),
        )

        model.save_checkpoint(CKPT_DIR / "last.pt", **shared_ckpt_kwargs)

        is_best = False
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_epoch = epoch
            epochs_no_improve = 0
            is_best = True
            model.save_checkpoint(CKPT_DIR / "best_top1.pt", **shared_ckpt_kwargs)
            log.info(f"  ★ New best Top-1: {best_top1:.4f}")
        else:
            epochs_no_improve += 1

        if metrics["gap"] > best_gap:
            best_gap = metrics["gap"]
            model.save_checkpoint(CKPT_DIR / "best_gap.pt", **shared_ckpt_kwargs)
            log.info(f"  ★ New best Gap:   {best_gap:.4f}")

        if epochs_no_improve >= patience:
            log.info(
                f"Early stopping triggered! No improvement in Top-1 for {patience} epochs."
            )
            break

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("TRAINING COMPLETE")
    log.info(f"  Best Top-1 : {best_top1:.4f} (Epoch {best_epoch})")
    log.info(f"  Best Gap   : {best_gap:.4f}")
    log.info(f"  Metrics    : {metrics_path}")
    log.info(f"  Checkpoints: {CKPT_DIR}")
    log.info("=" * 65)
    log.info("Comparison vs frozen baseline:")
    log.info(f"  Frozen  Top-1=0.9883  Gap=0.3773")
    log.info(f"  Tuned   Top-1={best_top1:.4f}  Gap={best_gap:.4f}")
    log.info(f"  Delta   Top-1={best_top1 - 0.9883:+.4f}  Gap={best_gap - 0.3773:+.4f}")
    log.info("=" * 65)

    # Call plotting script
    import subprocess

    try:
        plot_script = Path(__file__).parent / "plot_metrics.py"
        if plot_script.exists():
            log.info("Generating loss curves...")
            subprocess.run(["uv", "run", "python", str(plot_script)], check=True)
            log.info(f"Loss curves saved to {RESULTS_DIR}/loss_curves.png")
    except Exception as e:
        log.warning(f"Could not generate loss curves: {e}")


if __name__ == "__main__":
    main()
