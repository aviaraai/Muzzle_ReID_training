"""
scripts/train_dinov2_arcface.py — Fine-tune DINOv2 + Sub-center ArcFace

Two stages, two arms, selected by flag.

  --stage pretrain   folder-scans the 300-identity corpus (~2,460 images,
                     median 10 per identity), 45 held-out identities
  --stage finetune   the Uttarakhand manifest; refuses to start unless its
                     split_hash is the locked 1574f9bc...

  --arm full         518 full frames (~0.7 mm/px on the muzzle -- a 1mm ridge
                     gets ~1.4px, below Nyquist for the finest structure)
  --arm crop         518 muzzle crops (0.232 mm/px, ~4.3px per 1mm ridge).
                     Self-consistent: train, val and gallery all cropped, and
                     the run aborts if any input is not, because a mixed arm
                     reproduces the measured -22 point collapse.

Both splits are identity-DISJOINT and assert it on startup. That mirrors
production, which matches against animals the encoder never saw, and is why
evaluate() does leave-one-out retrieval within the query set rather than
querying it against the gallery.

Freeze schedule (the authority is _FREEZE_SCHEDULE in model.py, not this
comment — keep the two in step if you change it):
  Epoch  1–5  : Block 11    + LayerNorm + Head + GeM   ( 7.6M trainable)
  Epoch  6–20 : Blocks 9–11 + LayerNorm + Head + GeM   (21.8M)
  Epoch 21–45 : Blocks 6–11 + LayerNorm + Head + GeM   (~43M)
  Epoch 46–80 : Blocks 4–11 + LayerNorm + Head + GeM   (~57M)

Saves:
  checkpoints/phase_epochNN.pt — one per freeze-phase boundary (5/20/45/80),
                     kept so a failed frequency probe can be traced to the
                     phase where texture was still being read
  checkpoints/last.pt        — every epoch; the --resume point, not a deliverable
  checkpoints/best_top1.pt   — best retrieval Top-1 so far
  checkpoints/best_gap.pt    — best genuine/impostor separation so far
  results/metrics.csv, results/training.log, results/loss_curves.png

Each checkpoint carries optimizer state, so --resume is only valid across
runs of the SAME code: if the set of parameters handed to the optimizer
changes, an older checkpoint will not load. Delete checkpoints/ instead.

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
from model import GodhaarModel, build_model, _FREEZE_SCHEDULE
from augment import build_train_transform, build_val_transform
from corpus import (REQUIRED_SPLIT_HASH, load_clusters, load_folder_corpus,
                    load_manifest_corpus)
from sampler import PKClusterSampler
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

# Epochs at which a freeze phase ends -- checkpointed for the texture-vs-
# appearance post-mortem described at the save site below.
_PHASE_BOUNDARY_EPOCHS = {end for _s, end, _b in _FREEZE_SCHEDULE}

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
    """Backed by a corpus.Corpus, so the same class serves both the
    folder-scanned 300-identity corpus and the manifest-described Uttarakhand
    split without the training loop knowing which it got."""

    def __init__(self, corpus, transform):
        self.corpus = corpus
        self.transform = transform
        self.label_map = corpus.label_map

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        img_path = self.corpus.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.corpus.labels[idx]
        # idx is returned alongside (img, label) so hard-negative injection
        # can key off each sample's REAL position in this dataset -- see the
        # training loop's comment on why that must not be reconstructed from
        # a shuffled DataLoader's step number.
        return img, label, idx


# =============================================================================
# EVALUATION — fully vectorised (matmul, no Python loops)
# =============================================================================


@torch.no_grad()
def evaluate(model, loss_fn, gallery_loader, query_loader, device,
             identity_disjoint: bool | None = None):
    model.eval()

    gallery_embs, gallery_labels = [], []
    for imgs, lbls, _idxs in gallery_loader:
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            e = model(imgs.to(device))
        gallery_embs.append(e.cpu().numpy())
        gallery_labels.extend(lbls.tolist())
    gallery_embs = np.vstack(gallery_embs).astype(np.float32)
    gallery_labels = np.array(gallery_labels)

    query_embs, query_labels = [], []
    vl = 0.0
    for imgs, lbls, _idxs in query_loader:
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
    # The caller knows which case this is from the CORPORA themselves and must
    # say so. Inferring it here from integer labels was wrong and silently
    # destroyed a full 80-epoch A100 run: load_folder_corpus builds two
    # INDEPENDENT label spaces (train 0..230, val 0..44), so the val labels are
    # a strict subset of the train labels and `isdisjoint` returns False on a
    # split whose identities are in fact completely disjoint. The closed-set
    # branch then ran, scoring every val animal against the TRAIN gallery by
    # integer equality -- val animal 082 and train animal 039 both carry label
    # 7 and were counted as a genuine match. That produced top1=0.0082 (below
    # the 1/45=0.022 chance line) and gap=-0.0007 for a checkpoint that
    # actually scores top1=0.9429 under the correct protocol, early-stopped the
    # run at epoch 40 on a meaningless number, and left best_top1.pt pinned to
    # epoch 1.
    if identity_disjoint is None:
        identity_disjoint = set(gallery_labels.tolist()).isdisjoint(set(query_labels.tolist()))
        log.warning("  evaluate(): identity_disjoint not supplied, inferred %s from integer "
                    "labels -- unreliable when the corpora number their labels independently",
                    identity_disjoint)

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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune",
                        help="pretrain = folder-scan the 300-identity corpus; "
                             "finetune = the Uttarakhand manifest (split_hash enforced)")
    parser.add_argument("--arm", choices=["full", "crop"], default="full",
                        help="full = 518 full frames; crop = 518 muzzle crops, "
                             "self-consistent (train, val and gallery all cropped)")
    parser.add_argument("--corpus-dir", default=None,
                        help="root of the folder-scanned corpus (stage=pretrain)")
    parser.add_argument("--clusters", default=str(ROOT / "results" / "part3_appearance_clusters.json"))
    parser.add_argument("--val-identities", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--pk-p", type=int, default=8)
    parser.add_argument("--pk-k", type=int, default=4)
    parser.add_argument("--pk-same", type=int, default=6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--monitor", choices=["gap", "top1"], default="gap",
                        help="early-stopping metric. gap by default: the 17-epoch "
                             "run had top-1 flat from epoch 12 while separation "
                             "climbed to the final epoch")
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
        choices=["rtx3050", "rtx4060", "rtx4090", "a6000", "auto"],
        default="auto",
        help="GPU preset: auto-tunes batch-size, grad-accum and precision",
    )
    args = parser.parse_args()
    import sys as _sys
    args._explicit_lr = any(a.startswith("--lr-") for a in _sys.argv[1:])

    # ── GPU preset overrides ──────────────────────────────────────────────────
    AMP_DTYPE = "float16"  # default (safe for all GPUs)
    USE_COMPILE = False

    if args.gpu_preset == "auto":
        # detect Ada Lovelace (sm_89) or Ampere (sm_80+) automatically
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if cap[0] >= 8 and total_gb >= 40:
                args.gpu_preset = "a6000"  # Ampere/Ada, 40GB+ VRAM (A6000, A100, ...)
            elif cap[0] >= 8:
                args.gpu_preset = "rtx4060"  # treat other Ampere/Ada as high-end
            else:
                args.gpu_preset = "rtx3050"

    if args.gpu_preset == "a6000":
        # 48GB is far more than this ~330-image dataset needs, but a larger
        # batch means more distinct identities represented per ArcFace step,
        # which is a real win for a metric-learning loss on a small class
        # count -- there's no reason to leave the card underused here.
        if args.batch_size == 8 and args.grad_accum == 4:
            args.batch_size = 32
            args.grad_accum = 1
        AMP_DTYPE = "bfloat16"
        # torch.compile deliberately OFF here. Two reasons, both specific to
        # this run rather than general distrust of compile:
        #   1. Hard-negative injection (epoch 5 onward) appends a variable
        #      number of extra samples to each batch, so the batch dimension
        #      changes from step to step. compile guards on input shape, and
        #      mode="reduce-overhead" backs onto CUDA graphs, which is the
        #      configuration least tolerant of shifting shapes.
        #   2. This dataset is ~330 training images -- roughly 10 steps per
        #      epoch. Compilation overhead is paid against a few hundred
        #      steps total, so there is very little wall-clock left to win.
        # It also removes the Triton/C-compiler dependency inside the
        # container entirely, which is what broke the first A6000 run.
        USE_COMPILE = False
        log.info(
            "[GPU Preset: a6000] batch=%d  accum=%d  amp=bfloat16  compile=%s (off by design)",
            args.batch_size,
            args.grad_accum,
            USE_COMPILE,
        )
    elif args.gpu_preset in ("rtx4060", "rtx4090"):
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

    # Stage 2 is a domain adaptation, not a re-training: 439 images cannot
    # support deep unfreezing. Blocks 0-8 stay frozen throughout; only 9-11 +
    # LayerNorm + head + GeM move, at 10x lower LR than pretraining.
    if args.stage == "finetune" and not args._explicit_lr:
        args.lr_backbone, args.lr_head = 1e-5, 1e-4
        if args.epochs == 80:
            args.epochs = 30
        log.info("  Stage-2    : LR backbone=1e-5 head=1e-4, 30 epochs, blocks 0-8 frozen")

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

    # ── Corpus ───────────────────────────────────────────────────────────────
    # Two on-disk formats, one interface. Stage 1 folder-scans the 300-identity
    # corpus; stage 2 reads the Uttarakhand manifest and refuses to start
    # unless its split_hash is the locked one. Both assert identity-disjointness
    # internally and abort rather than train on a leaky split.
    if args.stage == "pretrain":
        corpus_dir = Path(args.corpus_dir) if args.corpus_dir else ROOT / "data"
        log.info(f"  Stage      : PRETRAIN (folder scan)  {corpus_dir}")
        train_corpus, val_corpus = load_folder_corpus(
            corpus_dir, n_val_identities=args.val_identities, seed=args.seed)
    else:
        log.info(f"  Stage      : FINETUNE (manifest)  {MANIFEST}")
        train_corpus, val_corpus = load_manifest_corpus(
            MANIFEST, DATA_ROOT, require_split_hash=REQUIRED_SPLIT_HASH)

    # Decided HERE, from the identity strings, because this is the only place
    # that still has them -- evaluate() sees integer labels only, and the two
    # corpora number their labels independently from 0, so integers cannot
    # answer this question. Picks the retrieval protocol: open-set leave-one-out
    # within the val set (identities disjoint) vs closed-set search against the
    # train gallery (identities shared).
    split_is_identity_disjoint = set(train_corpus.identities).isdisjoint(
        set(val_corpus.identities))

    num_classes = train_corpus.num_classes
    log.info(f"  Arm        : {args.arm}")
    log.info(f"  Classes    : {num_classes}  (train identities)")
    log.info(f"  Train      : {len(train_corpus):,} images / {num_classes} identities")
    log.info(f"  Val        : {len(val_corpus):,} images / {val_corpus.num_classes} identities "
             f"(identity-disjoint, asserted)")
    log.info(f"  Protocol   : {'OPEN-SET leave-one-out within val' if split_is_identity_disjoint else 'CLOSED-SET search vs train gallery'}")

    # Arm 2 is only meaningful if EVERY input is cropped -- train, val and the
    # gallery used for retrieval alike. A mixed arm reproduces the measured
    # -22 point collapse inside the experiment and would be indistinguishable
    # from the texture hypothesis failing.
    #
    # Checked via a marker file at the corpus root (written by
    # make_arm2_crops.py), NOT a substring in the file path. A path check
    # breaks the instant the corpus is bind-mounted somewhere else -- Docker
    # remaps whatever host folder you point DATA_DIR at to /workspace/data,
    # so a host folder literally named "300-crops-arm2" loses the one
    # substring the old check looked for. A marker file survives any mount
    # point, rename, or uv-direct invocation with no container at all.
    if args.arm == "crop":
        marker = Path(args.corpus_dir or (ROOT / "data")) / "_ARM2_CROP_MARKER.json"
        if not marker.exists():
            raise SystemExit(
                f"ABORT: --arm crop requires {marker} to exist. Its absence means this "
                "corpus was not built by make_arm2_crops.py, or --corpus-dir points "
                "somewhere else. A mixed cropped/uncropped arm silently reproduces the "
                "-22 point collapse measured earlier and would invalidate the experiment. "
                "If this corpus genuinely is all crops, create the marker: "
                f'python -c "import json,pathlib; pathlib.Path(r\'{marker}\').write_text('
                'json.dumps({\\"crop_frac\\": 0.60, \\"verified_manually\\": true}))"'
            )
        log.info(f"  Arm check  : {marker.name} present -> all inputs cropped OK")

    # ── Transforms ───────────────────────────────────────────────────────────
    # Augmentation policy -- each choice tied to a measurement, not a guess
    # (re-verified against the real production quality/crop gate, not the
    # unreliable subjective visual-review pass this dataset's prior audit
    # turned out to have -- see rebuild_manifest.py's docstring for what was
    # wrong with that and why it was rebuilt on objective criteria alone):
    #
    #   - RandomResizedCrop(scale, ratio): simulates the real crop-aspect
    #     spread this pipeline injects upstream of the encoder (handheld
    #     capture distance/angle varies a lot between an animal's own
    #     photos) -- ratio=(0.6, 1.5) matches the measured raw-file aspect
    #     range (0.62-1.59, re-confirmed directly on this exact manifest).
    #   - NO RandomHorizontalFlip. Muzzle bead/ridge patterns are not
    #     symmetric -- flipping risks synthesizing a pattern that looks like
    #     a different real identity, manufacturing a false negative signal
    #     for ArcFace to learn from. (The un-audited original script had
    #     this at p=0.5; removed.)
    #   - NO RandomPerspective. Not tied to any real measurement of this
    #     dataset's capture geometry -- dropped rather than kept on a guess.
    #   - GaussianBlur: real field photos span a wide sharpness range (the
    #     production quality gate's own floor is BLUR_THRESHOLD=20.0
    #     Laplacian variance, and passing images still range into the
    #     hundreds) -- training should see some near-floor examples, not
    #     just sharp ones.
    #   - ColorJitter: mild general lighting/coat-color robustness. hue
    #     jitter kept small (cattle coat/muzzle color IS part of what
    #     distinguishes individuals within a breed, even though it's not
    #     the primary re-id signal -- don't scramble it).
    #   - RandomErasing at a LARGER patch size and higher probability than
    #     the original script: real photos routinely have a handler's hand,
    #     a rope/halter, or feed/hay partially covering the muzzle. Bigger,
    #     more frequent random erasing teaches the encoder not to depend on
    #     any single small patch being visible.
    # Augmentation lives in augment.py so the removals are documented next to
    # the measurements that justify them. GaussianBlur(sigma<=3.0) and
    # RandomResizedCrop(ratio 0.6-1.5) are GONE -- the first destroys the band
    # this experiment measures, the second manufactures aspect distortion far
    # outside the observed [0.78, 1.09].
    train_transform = build_train_transform(IMG_SIZE)
    val_transform = build_val_transform(IMG_SIZE)
    log.info("  Augment    : " + ", ".join(type(t).__name__ for t in train_transform.transforms))

    train_dataset = MuzzleDataset(train_corpus, train_transform)
    gallery_dataset = MuzzleDataset(train_corpus, val_transform)
    val_dataset = MuzzleDataset(val_corpus, val_transform)

    # P x K sampling with appearance-cluster hard negatives. Without it a
    # batch of eight differently-coloured animals is separable on colour alone
    # and the model never has to read the print -- which is the entire point
    # of the pretraining run.
    clusters = load_clusters(Path(args.clusters), set(train_corpus.identities))
    sampler = PKClusterSampler(
        train_corpus.indices_by_identity(), clusters,
        P=args.pk_p, K=args.pk_k, same_cluster=args.pk_same, seed=args.seed)
    log.info(f"  Sampler    : P={args.pk_p} x K={args.pk_k} = {args.pk_p*args.pk_k}/batch, "
             f"{args.pk_same} of {args.pk_p} identities from one appearance cluster, "
             f"{len(sampler)} batches/epoch")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
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

    # ── Auto-resume ──────────────────────────────────────────────────────────
    # If the run stops for any reason (killed, machine reboot, OOM) and gets
    # restarted with the SAME command, it should not silently repeat epochs
    # already paid for on an A6000. Detect an existing last.pt and resume
    # automatically rather than requiring --resume to be remembered.
    #
    # This is safe only because it is guarded: checkpoints/ is a single
    # directory shared across every stage/arm combination, so a bare "does
    # last.pt exist" check would happily resume a finetune run onto a
    # pretrain checkpoint, or a crop-arm run onto a full-arm one -- wrong
    # num_classes, wrong optimizer param groups, wrong everything. Compare
    # the checkpoint's own recorded stage/arm/class-count against this
    # invocation first; a mismatch aborts instead of corrupting the run.
    _last_ckpt_path = CKPT_DIR / "last.pt"
    if not args.resume and _last_ckpt_path.exists():
        _peek = torch.load(_last_ckpt_path, map_location="cpu", weights_only=False)
        _match = (_peek.get("stage") == args.stage and _peek.get("arm") == args.arm
                 and _peek.get("config", {}).get("num_classes") == num_classes)
        if _match:
            log.warning(f"  {_last_ckpt_path} found (epoch {_peek.get('epoch')}, same "
                        f"stage/arm/classes) -- AUTO-RESUMING. Delete checkpoints/ first "
                        f"for a genuinely fresh run.")
            args.resume = True
        else:
            raise SystemExit(
                f"ABORT: {_last_ckpt_path} exists but belongs to a different run "
                f"(stage={_peek.get('stage')!r} arm={_peek.get('arm')!r} "
                f"classes={_peek.get('config', {}).get('num_classes')}) than this "
                f"invocation (stage={args.stage!r} arm={args.arm!r} classes={num_classes}). "
                f"Auto-resuming across a mismatch would load the wrong architecture. "
                f"Pass --resume to force it, or delete checkpoints/ for a fresh run."
            )
        del _peek

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

    # The invariant that matters most in this file. parameter_groups() must
    # hand AdamW the WHOLE backbone, including the parts the freeze schedule
    # has not opened yet -- otherwise every phase past the first silently
    # trains nothing. That regression shipped an untrained backbone to
    # production for months (fix 15a9115, pinned by
    # scripts/test_training.py::test_optimizer_holds_every_backbone_param_at_construction).
    _in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    _bb_held = sum(p.numel() for p in model.backbone.parameters() if id(p) in _in_opt)
    _bb_total = sum(p.numel() for p in model.backbone.parameters())
    if _bb_held != _bb_total:
        raise SystemExit(
            f"ABORT: optimizer holds {_bb_held:,} of {_bb_total:,} backbone params. "
            "parameter_groups() is filtering on requires_grad again -- the freeze "
            "schedule would be a silent no-op. Refusing to train."
        )
    log.info(f"  Optimizer  : holds {_bb_held:,}/{_bb_total:,} backbone params "
             f"(all phases will actually train)")

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

    patience = args.patience
    epochs_no_improve = 0
    best_epoch = start_epoch - 1
    last_unfreeze_from = None

    log.info("\nStarting training...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        unfreeze_from = model.progressive_unfreeze(epoch)

        # Reset the early-stopping counter whenever the freeze schedule
        # unfreezes MORE of the backbone. Without this, a plateau reached
        # under a MORE restrictive freeze state (fewer trainable params --
        # only 8.7% of the model during epochs 1-15, per the freeze
        # schedule in godhaar/model.py) can trigger early stopping at
        # patience=15 BEFORE the model ever gets access to the deeper
        # blocks the schedule was going to unfreeze at epoch 16 -- on a
        # dataset this small (331 train images), where the block-11-only
        # phase can plausibly saturate quickly, that means blocks 9-11 and
        # 6-11 might never be touched at all, silently. A prior plateau
        # under less capacity says nothing about what happens once the
        # model gets more of it, so it must not count against a freshly
        # more-capable model.
        if last_unfreeze_from is not None and unfreeze_from != last_unfreeze_from:
            if epochs_no_improve > 0:
                log.info(
                    f"  Freeze schedule advanced (unfreeze_from {last_unfreeze_from}->{unfreeze_from}) "
                    f"-- resetting early-stopping counter (was {epochs_no_improve}/{patience})"
                )
            epochs_no_improve = 0
        last_unfreeze_from = unfreeze_from

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

        for step, (imgs, labels, idxs) in enumerate(train_loader):
            # Inject hard negatives, keyed to the REAL dataset position of
            # each anchor actually in this batch (idxs, from MuzzleDataset --
            # train_loader has shuffle=True, so `step*batch_size` is NOT the
            # position range of what's in this batch; using it to index
            # hard_negs was silently injecting negatives keyed to whichever
            # samples happened to occupy those positions before shuffling,
            # unrelated to this batch's actual anchors -- found by tracing
            # this end to end, not by running it, since it never raised or
            # produced an obviously-wrong loss value).
            if hard_negs:
                extra_imgs, extra_labels = [], []
                for idx in idxs.tolist():
                    pool = hard_negs.get(idx, [])
                    if pool:
                        ni = random.choice(pool)
                        neg_img, neg_label, _ = gallery_dataset[ni]
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
            model, loss_fn, gallery_loader, query_loader, device,
            identity_disjoint=split_is_identity_disjoint,
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
            # Recorded so a later run can tell whether checkpoints/last.pt is
            # safe to auto-resume onto -- see the auto-resume guard above.
            stage=args.stage,
            arm=args.arm,
        )

        model.save_checkpoint(CKPT_DIR / "last.pt", **shared_ckpt_kwargs)

        # Freeze-phase boundaries. If the frequency probe later says the
        # encoder reads appearance rather than texture, these are the only way
        # to tell whether an EARLIER phase read texture before deeper
        # unfreezing let it drift back -- a diagnostic that cannot be
        # reconstructed after the fact.
        if epoch in _PHASE_BOUNDARY_EPOCHS:
            model.save_checkpoint(CKPT_DIR / f"phase_epoch{epoch:02d}.pt", **shared_ckpt_kwargs)
            log.info(f"  → freeze-phase boundary checkpoint: phase_epoch{epoch:02d}.pt")

        is_best = False
        improved_top1 = metrics["top1"] > best_top1
        improved_gap = metrics["gap"] > best_gap
        # The early-stopping counter tracks --monitor (separation by default);
        # both checkpoints are still written on their own metric.
        if (improved_gap if args.monitor == "gap" else improved_top1):
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if improved_top1:
            best_top1 = metrics["top1"]
            best_epoch = epoch
            is_best = True
            model.save_checkpoint(CKPT_DIR / "best_top1.pt", **shared_ckpt_kwargs)
            log.info(f"  ★ New best Top-1: {best_top1:.4f}")

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
    # NOTE: the reference number to compare this against is NOT hardcoded
    # here on purpose -- 0.9883/0.3773 (removed) was the ORIGINAL foreign
    # 300-cattle checkpoint's own held-out-IMAGE metric (same identities in
    # train and val, a much easier and structurally different task than this
    # run's identity-DISJOINT open-set split). Compare best_top1 above
    # against the real reference points instead: the raw current-production
    # checkpoint's identity-disjoint LOO baseline (58.7% top-1, 41.3%
    # impostor-beats-genuine, this exact Godhaar_aron corpus) and the
    # separately-measured fusion+whitening pipeline's numbers -- not a
    # number from a different dataset and a different (easier) protocol.
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
