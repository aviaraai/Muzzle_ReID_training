"""Godhaar AI — Training losses (separate from the encoder).

ArcFace and any future metric-learning losses live here, not inside
GodhaarModel.  This keeps the encoder exportable (ONNX / TorchScript)
without carrying training-only state.

Usage
-----
    from model  import build_model
    from losses import build_arcface, arcface_parameter_groups

    encoder  = build_model(num_classes=300, device=device)
    loss_fn  = build_arcface(num_classes=300, device=device)

    # training step
    emb  = encoder(images)                        # (B, 256)
    loss = loss_fn(emb.float(), labels)           # cast to fp32 for ArcFace

    # optimizer covers both encoder and arcface weights
    opt = torch.optim.AdamW([
        *encoder.parameter_groups(),
        *arcface_parameter_groups(loss_fn, lr=1e-3, weight_decay=0.01),
    ])

    # checkpoint: save encoder and arcface separately
    encoder.save_checkpoint("ckpt/encoder.pt", epoch=epoch, metrics=metrics)
    save_arcface(loss_fn, "ckpt/arcface.pt", epoch=epoch)

Mixed precision note
--------------------
ArcFace's cosine similarity can underflow in fp16.  Always pass
``embeddings.float()`` (fp32) to the loss, even when the encoder
forward runs under ``torch.amp.autocast()``.

Author: Godhaar AI Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_metric_learning.losses import SubCenterArcFaceLoss

log = logging.getLogger("godhaar.losses")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(_h)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_EMB_DIM: int          = 256
_ARCFACE_SCALE: float  = 64.0
_ARCFACE_MARGIN: float = 0.50   # ≈ 28.6°
_ARCFACE_K: int        = 3      # sub-centers — robust to label noise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_arcface(
    num_classes: int,
    embedding_size: int   = _EMB_DIM,
    scale: float          = _ARCFACE_SCALE,
    margin: float         = _ARCFACE_MARGIN,
    sub_centers: int      = _ARCFACE_K,
    device: str | torch.device = "cpu",
) -> SubCenterArcFaceLoss:
    """Construct a Sub-center ArcFace loss and move it to ``device``.

    Parameters
    ----------
    num_classes : int
        Number of cattle identities.
    embedding_size : int
        Must match the encoder's output dimension (256).
    scale : float
        ArcFace scale (temperature) ``s``.
    margin : float
        ArcFace angular margin ``m`` in radians.
    sub_centers : int
        Number of sub-centers per class (robustness to label noise).
    device : str or torch.device

    Returns
    -------
    SubCenterArcFaceLoss  moved to ``device``.
    """
    loss_fn = SubCenterArcFaceLoss(
        num_classes     = num_classes,
        embedding_size  = embedding_size,
        margin          = margin,
        scale           = scale,
        sub_centers     = sub_centers,
    ).to(device)

    n_params = sum(p.numel() for p in loss_fn.parameters())
    log.info(
        f"SubCenterArcFace: classes={num_classes}, emb={embedding_size}, "
        f"K={sub_centers}, s={scale}, m={margin:.4f}  "
        f"params={n_params:,}"
    )
    return loss_fn


def arcface_parameter_groups(
    loss_fn: SubCenterArcFaceLoss,
    lr: float           = 1e-3,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Return AdamW parameter groups for the ArcFace weight matrix.

    Parameters
    ----------
    loss_fn : SubCenterArcFaceLoss
    lr : float
        Should match the encoder head LR (``lr_head``).
    weight_decay : float

    Returns
    -------
    List[dict]  — two groups (wd / no-wd) for AdamW.
    """
    decay, no_decay = [], []
    for name, param in loss_fn.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "lr": lr, "weight_decay": weight_decay,
         "name": "arcface_wd"},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0,
         "name": "arcface_no_wd"},
    ]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_arcface(
    loss_fn: SubCenterArcFaceLoss,
    path: str | Path,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    **extra: Any,
) -> None:
    """Save ArcFace weights to disk.

    Parameters
    ----------
    loss_fn : SubCenterArcFaceLoss
    path : str or Path
    epoch : int
    metrics : dict, optional
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":               epoch,
            "loss_fn_state_dict":  loss_fn.state_dict(),
            "metrics":             metrics or {},
            **extra,
        },
        path,
    )
    size_mb = path.stat().st_size / 1e6
    log.info(f"ArcFace checkpoint saved → {path}  ({size_mb:.1f} MB, epoch={epoch})")


def load_arcface(
    loss_fn: SubCenterArcFaceLoss,
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load ArcFace weights into an existing loss_fn instance.

    Parameters
    ----------
    loss_fn : SubCenterArcFaceLoss
        Must have been constructed with matching num_classes / embedding_size.
    path : str or Path
    device : str or torch.device
    strict : bool

    Returns
    -------
    dict  — full checkpoint (epoch, metrics, …)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ArcFace checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    loss_fn.load_state_dict(ckpt["loss_fn_state_dict"], strict=strict)
    loss_fn.to(device)
    log.info(
        f"ArcFace weights loaded from {path}  "
        f"(epoch={ckpt.get('epoch', '?')})"
    )
    return ckpt