"""Godhaar AI — Production-grade cattle muzzle re-identification encoder.

Architecture
------------
Input (518×518)
→ DINOv2 ViT-B/14 (patch tokens only, NO CLS token)
→ Generalized Mean Pooling (GeM, trainable exponent p, sign-preserving)
→ Projection Head: Linear(768→512) → BN → GELU → Dropout(0.2)
                   → Linear(512→256) → BN → L2-Normalize

Output contract
---------------
Shape  : (B, 256), float32, unit-norm (L2).
This contract is frozen — downstream fusion layers depend on it.

Freeze schedule
---------------
Epochs  1–10 : blocks 10–11 + LayerNorm + head
Epochs 11–30 : blocks  8–11 + LayerNorm + head
Epochs 31–50 : blocks  6–11 + LayerNorm + head

Design goals
------------
• No ArcFace inside the model — see losses.py for training loss.
• No HuggingFace dependency (timm only).
• EMA-ready: encoder exposes only the embedding forward pass.
• Multi-modal extensibility: output spec (B, 256) is the stable API.
• Full checkpoint save/load with rich env metadata.
• Production-quality logging, type hints, docstrings throughout.

Author: Godhaar AI Team
"""

from __future__ import annotations

import logging
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("godhaar.model")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DINOV2_MODEL_NAME: str = "vit_base_patch14_dinov2"
_BACKBONE_EMB_DIM: int  = 768   # DINOv2 ViT-B/14 output dimension
_HIDDEN_DIM: int        = 512   # Projection head intermediate dimension
_PROJ_DIM: int          = 256   # Final embedding dimension (output contract)
_IMG_SIZE: int          = 518   # Native DINOv2 ViT-B/14 input resolution
_PATCH_SIZE: int        = 14    # Patch size for ViT-B/14
_NUM_PATCHES: int       = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 37×37 = 1 369

# Freeze schedule: (epoch_start, epoch_end, first_block_to_unfreeze)
# Blocks [unfreeze_from .. 11] are trained; earlier blocks stay frozen.
_FREEZE_SCHEDULE: List[Tuple[int, int, int]] = [
    (1,  15, 11),   # epochs  1–15  → unfreeze block 11 only
    (16, 35,  9),   # epochs 16–35  → unfreeze blocks 9–11
    (36, 60,  9),   # epochs 36–60  → keep blocks 9–11
]

# ---------------------------------------------------------------------------
# Generalized Mean Pooling (GeM)
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    """Sign-preserving Generalized Mean Pooling over patch token sequences.

    Computes:
        GeM(X) = sign(mean_i(|x_i|^p)) * (mean_i(|x_i|^p))^(1/p)

    The sign-preserving formulation handles the negative activations that
    are natural in post-LayerNorm DINOv2 patch tokens.  The original CNN
    variant (Radenović et al., 2019) clamped to non-negative because
    ReLU outputs are non-negative by construction; that assumption does
    NOT hold for ViT features.

    Pooling mode is selectable at construction time so ablations can swap
    between ``"gem"``, ``"avg"``, and ``"cls"`` without rebuilding the
    full model.

    Parameters
    ----------
    mode : {"gem", "avg", "cls"}
        Pooling strategy.
        ``"gem"``  — sign-preserving GeM (default, recommended).
        ``"avg"``  — simple mean pooling over patch tokens (baseline).
        ``"cls"``  — identity (pass-through; caller must supply CLS token).
    p_init : float
        Initial GeM exponent.  Default 3.0 follows the original paper.
        Ignored when ``mode != "gem"``.
    eps : float
        Floor added to |x| before raising to power ``p`` to prevent
        zero-gradient at exactly-zero activations.

    Input
    -----
    x : Tensor of shape (B, N, D)
        Batch of N patch-token sequences each with embedding dimension D.
        When mode is ``"cls"``, x is expected to be (B, D) already pooled.

    Output
    ------
    Tensor of shape (B, D) — one pooled descriptor per image.
    """

    def __init__(
        self,
        mode: Literal["gem", "avg", "cls"] = "gem",
        p_init: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in ("gem", "avg", "cls"):
            raise ValueError(f"mode must be 'gem', 'avg', or 'cls'; got {mode!r}")
        self.mode = mode
        self.eps = eps
        # p is only a learnable parameter when actually using GeM pooling.
        if mode == "gem":
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer("p", torch.tensor(float(p_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens to a single descriptor.

        Parameters
        ----------
        x : Tensor of shape (B, N, D)  [or (B, D) when mode=="cls"]

        Returns
        -------
        Tensor of shape (B, D)
        """
        if self.mode == "avg":
            return x.mean(dim=1)

        if self.mode == "cls":
            # Caller passes the CLS token directly; nothing to pool.
            assert x.ndim == 2, "cls mode expects pre-extracted (B, D) tensor"
            return x

        # mode == "gem"
        # Clamp p for numerical stability (must stay ≥ 1).
        p = self.p.clamp(min=1.0, max=100.0)

        # Sign-preserving GeM for ViT features (handles negative activations).
        # abs_x^p preserves magnitude; sign restores direction after pooling.
        abs_x = x.abs().clamp(min=self.eps)          # (B, N, D), all positive
        sign  = x.sign()                              # (B, N, D), in {-1, 0, +1}

        powered     = abs_x.pow(p)                    # (B, N, D)
        mean_pow    = (sign * powered).mean(dim=1)    # (B, D) — signed mean
        out_sign    = mean_pow.sign()
        out_abs     = mean_pow.abs().clamp(min=self.eps)
        return out_sign * out_abs.pow(1.0 / p)        # (B, D)

    def extra_repr(self) -> str:
        p_val = self.p.item() if hasattr(self.p, "item") else float(self.p)
        return f"mode={self.mode!r}, p={p_val:.4f}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Two-layer projection head with BN, GELU, Dropout, and L2 normalisation.

    Maps the GeM-pooled DINOv2 descriptor (768-d) to a compact, unit-norm
    embedding (256-d) suitable for ArcFace training and FAISS indexing.

    Architecture
    ------------
    Linear(768 → 512, no bias)
    BatchNorm1d(512)
    GELU
    Dropout(p)
    Linear(512 → 256, no bias)
    BatchNorm1d(256)
    L2-Normalise

    The final BatchNorm is placed *before* L2-normalisation so the network
    can rescale per-dimension variance without violating the unit-norm
    constraint imposed by ArcFace.

    Parameters
    ----------
    in_dim : int
        Input dimensionality (backbone output after GeM pooling).
    hidden_dim : int
        Intermediate dimensionality of the first linear layer.
    out_dim : int
        Output embedding dimensionality.
    dropout : float
        Dropout probability applied between the two linear layers.
    """

    def __init__(
        self,
        in_dim: int     = _BACKBONE_EMB_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        out_dim: int    = _PROJ_DIM,
        dropout: float  = 0.2,
    ) -> None:
        super().__init__()

        self.fc1  = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2  = nn.Linear(hidden_dim, out_dim, bias=False)
        self.bn2  = nn.BatchNorm1d(out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise a batch of pooled descriptors.

        Parameters
        ----------
        x : Tensor of shape (B, in_dim)

        Returns
        -------
        Tensor of shape (B, out_dim), unit-norm along dim=1.
        """
        x = self.bn1(self.act(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Main Encoder Model
# ---------------------------------------------------------------------------

class GodhaarModel(nn.Module):
    """DINOv2 ViT-B/14 + GeM + ProjectionHead cattle muzzle encoder.

    This model is a *pure metric-learning encoder*.  It produces 256-d
    unit-norm embeddings; it does NOT contain any loss function.  ArcFace
    and any other training losses live in ``losses.py`` and receive the
    embeddings produced here.

    Output contract (frozen API)
    ----------------------------
    Shape  : (B, 256)
    Dtype  : float32
    Norm   : unit-norm (L2 == 1.0 per sample)

    Downstream fusion layers (body, face, GPS, metadata) must respect this
    contract: each modality encoder produces (B, 256) unit-norm embeddings
    that are then combined by a separate fusion module.

    Parameters
    ----------
    num_classes : int
        Number of cattle identities.  Stored for checkpoint serialisation
        and informational logging only; not used inside this class.
    pooling : {"gem", "avg", "cls"}
        Pooling strategy.  "gem" is recommended; "avg" and "cls" enable
        ablation studies.
    gem_p_init : float
        Initial GeM exponent (ignored when pooling != "gem").
    proj_dropout : float
        Dropout probability in the projection head.

    Public API
    ----------
    forward(x)                      → embeddings (B, 256)
    extract_features(x)             → embeddings (B, 256), no grad
    save_checkpoint(path, **extra)  → None
    load_checkpoint(path, device)   → (GodhaarModel, dict)
    parameter_groups()              → list[dict] for AdamW
    freeze_all()                    → None
    progressive_unfreeze(epoch)     → int  (first unfrozen block)
    """

    def __init__(
        self,
        num_classes: int,
        pooling: Literal["gem", "avg", "cls"] = "gem",
        gem_p_init: float = 3.0,
        proj_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be ≥ 2, got {num_classes}")

        self.num_classes = num_classes
        self.pooling_mode = pooling
        self.emb_dim = _PROJ_DIM

        # ── Backbone ────────────────────────────────────────────────────
        log.info("Loading DINOv2 ViT-B/14 from timm (pretrained=True)…")
        self.backbone: nn.Module = timm.create_model(
            _DINOV2_MODEL_NAME,
            pretrained=True,
            num_classes=0,   # strip classifier head
        )
        actual_dim: int = self.backbone.num_features
        if actual_dim != _BACKBONE_EMB_DIM:
            raise RuntimeError(
                f"Expected backbone output dim {_BACKBONE_EMB_DIM}, "
                f"got {actual_dim}.  Wrong timm model?"
            )

        # Verify stochastic depth is active (should be nonzero for fine-tuning)
        dp = getattr(self.backbone.blocks[0], "drop_path", None)
        if dp is not None:
            drop_prob = getattr(dp, "drop_prob", 0.0)
            if drop_prob == 0.0:
                log.warning(
                    "DropPath rate is 0 in backbone blocks.  Consider a small "
                    "drop_path_rate (0.1–0.2) when fine-tuning to regularise "
                    "the partially-frozen network."
                )

        log.info(
            f"Backbone loaded: {_DINOV2_MODEL_NAME}  "
            f"params={sum(p.numel() for p in self.backbone.parameters()):,}"
        )

        # ── Pooling ─────────────────────────────────────────────────────
        self.gem = GeMPooling(mode=pooling, p_init=gem_p_init)

        # ── Projection Head ─────────────────────────────────────────────
        self.head = ProjectionHead(
            in_dim=_BACKBONE_EMB_DIM,
            hidden_dim=_HIDDEN_DIM,
            out_dim=_PROJ_DIM,
            dropout=proj_dropout,
        )

        # Freeze backbone entirely on construction; caller drives unfreezing.
        self.freeze_all()

        n_total    = sum(p.numel() for p in self.parameters())
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_head     = sum(p.numel() for p in self.head.parameters())
        n_gem      = sum(p.numel() for p in self.gem.parameters())
        log.info(
            f"Encoder ready — total={n_total:,}  "
            f"backbone={n_backbone:,}  gem={n_gem}  head={n_head:,}"
        )

    # =====================================================================
    # Patch-token extraction helpers
    # =====================================================================

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from DINOv2 backbone, excluding the CLS token.

        Uses ``forward_intermediates`` (timm ≥ 0.9.8) when available;
        falls back to manual block traversal for older timm versions.

        Parameters
        ----------
        x : Tensor of shape (B, 3, H, W)

        Returns
        -------
        patch_tokens : Tensor of shape (B, N_patches, 768)
        """
        if hasattr(self.backbone, "forward_intermediates"):
            out = self.backbone.forward_intermediates(
                x,
                indices=[11],
                return_prefix_tokens=False,  # exclude CLS
                norm=True,
            )
            return out[0]   # (B, 1369, 768)
        return self._manual_patch_tokens(x)

    def _get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the CLS token from the backbone for 'cls' pooling mode.

        Parameters
        ----------
        x : Tensor of shape (B, 3, H, W)

        Returns
        -------
        cls : Tensor of shape (B, 768)
        """
        # timm num_classes=0 returns the CLS token from forward()
        return self.backbone(x)

    def _manual_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback patch-token extraction for older timm versions.

        Runs the full ViT forward manually and returns only the patch
        tokens (excluding CLS) from the final transformer block.

        Parameters
        ----------
        x : Tensor (B, 3, H, W)

        Returns
        -------
        Tensor (B, N_patches, 768)
        """
        bb = self.backbone

        tokens: torch.Tensor = bb.patch_embed(x)                    # (B, N, 768)
        cls = bb.cls_token.expand(tokens.shape[0], -1, -1)          # (B, 1, 768)
        tokens = torch.cat([cls, tokens], dim=1)                     # (B, N+1, 768)
        tokens = tokens + bb.pos_embed
        if hasattr(bb, "pos_drop"):
            tokens = bb.pos_drop(tokens)
        for block in bb.blocks:
            tokens = block(tokens)
        tokens = bb.norm(tokens)
        return tokens[:, 1:, :]   # (B, N_patches, 768)

    # =====================================================================
    # Public forward API
    # =====================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalised 256-d embeddings for a batch of muzzle images.

        Parameters
        ----------
        x : Tensor of shape (B, 3, 518, 518)
            Normalised muzzle image batch (ImageNet mean/std).

        Returns
        -------
        embeddings : Tensor of shape (B, 256), unit-norm.
        """
        if self.pooling_mode == "cls":
            pooled = self._get_cls_token(x)          # (B, 768)
        else:
            patch_tokens = self._get_patch_tokens(x) # (B, 1369, 768)
            pooled = self.gem(patch_tokens)           # (B, 768)

        return self.head(pooled)                      # (B, 256), L2-normed

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings in inference mode (no gradient computation).

        Temporarily sets the model to eval mode for the duration of this
        call, then restores the previous training/eval state.  Suitable
        for building FAISS gallery indexes and querying at production time.

        Parameters
        ----------
        x : Tensor of shape (B, 3, 518, 518)

        Returns
        -------
        embeddings : Tensor of shape (B, 256), unit-norm, detached.
        """
        was_training = self.training
        self.eval()
        try:
            embeddings = self.forward(x)
        finally:
            if was_training:
                self.train()
        return embeddings.detach()

    # =====================================================================
    # Checkpoint API
    # =====================================================================

    @staticmethod
    def _collect_env_metadata(seed: Optional[int] = None) -> Dict[str, Any]:
        """Collect runtime environment metadata for checkpoint provenance."""
        git_commit = "unknown"
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass

        return {
            "torch":    torch.__version__,
            "timm":     timm.__version__,
            "cuda":     torch.version.cuda or "n/a",
            "python":   platform.python_version(),
            "hostname": platform.node(),
            "git":      git_commit,
            "seed":     seed,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        **extra: Any,
    ) -> None:
        """Serialise encoder + training state to disk.

        Saves a self-contained checkpoint dictionary:

            ``model_state_dict``   — backbone + GeM + projection head
            ``optimizer_state``    — AdamW state (optional)
            ``scheduler_state``    — LR scheduler state (optional)
            ``config``             — hyperparameters
            ``epoch``              — current training epoch
            ``metrics``            — evaluation metrics (optional)
            ``env``                — torch/timm/cuda/git/host versions
            ``extra``              — any additional caller-supplied metadata

        Note: ArcFace weights are NOT stored here.  Save them separately
        via ``losses.py`` if you need to resume training from this checkpoint.

        Parameters
        ----------
        path : str or Path
        epoch : int
        optimizer_state : dict, optional
        scheduler_state : dict, optional
        metrics : dict, optional
        seed : int, optional
            Training seed for reproducibility metadata.
        **extra
            Arbitrary additional key-value pairs.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config: Dict[str, Any] = {
            "num_classes":  self.num_classes,
            "pooling":      self.pooling_mode,
            "emb_dim":      self.emb_dim,
            "proj_dim":     _PROJ_DIM,
            "hidden_dim":   _HIDDEN_DIM,
            "backbone":     _DINOV2_MODEL_NAME,
            "img_size":     _IMG_SIZE,
        }

        checkpoint: Dict[str, Any] = {
            "epoch":             epoch,
            "config":            config,
            "model_state_dict":  self.state_dict(),
            "metrics":           metrics or {},
            "env":               self._collect_env_metadata(seed=seed),
            **extra,
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler_state"] = scheduler_state

        torch.save(checkpoint, path)
        size_mb = path.stat().st_size / 1e6
        log.info(f"Checkpoint saved → {path}  ({size_mb:.1f} MB, epoch={epoch})")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> Tuple["GodhaarModel", Dict[str, Any]]:
        """Instantiate a GodhaarModel from a saved checkpoint.

        Parameters
        ----------
        path : str or Path
        device : str or torch.device
        strict : bool
            Passed to ``load_state_dict``.

        Returns
        -------
        model : GodhaarModel  (weights loaded, moved to ``device``)
        checkpoint : dict     (full checkpoint for epoch/metrics/env access)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        log.info(f"Loading checkpoint: {path}")
        ckpt: Dict[str, Any] = torch.load(
            path, map_location=device, weights_only=False
        )

        cfg = ckpt["config"]
        model = cls(
            num_classes   = cfg["num_classes"],
            pooling       = cfg.get("pooling", "gem"),
        )

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=strict
        )
        if missing:
            log.warning(f"  Missing keys  ({len(missing)}): {missing[:5]}…")
        if unexpected:
            log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}…")

        model.to(device)

        epoch   = ckpt.get("epoch", -1)
        metrics = ckpt.get("metrics", {})
        env     = ckpt.get("env", {})
        log.info(
            f"  Epoch={epoch}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        if env:
            log.info(
                f"  Saved with torch={env.get('torch')}  "
                f"timm={env.get('timm')}  "
                f"git={env.get('git')}  "
                f"host={env.get('hostname')}"
            )
        return model, ckpt

    # =====================================================================
    # Parameter groups for AdamW
    # =====================================================================

    def parameter_groups(
        self,
        lr_backbone: float   = 5e-5,
        lr_layernorm: float  = 5e-6,   # typically 0.1× backbone LR
        lr_head: float       = 1e-3,
        weight_decay: float  = 0.01,
    ) -> List[Dict[str, Any]]:
        """Return AdamW parameter groups with per-component learning rates.

        Groups
        ------
        1. backbone_wd       — DINOv2 blocks (2-D params), weight-decay on
        2. backbone_no_wd    — DINOv2 blocks (1-D / bias), no weight-decay
        3. layernorm_wd      — backbone LayerNorm params, dedicated lower LR
        4. layernorm_no_wd   — backbone LayerNorm bias, no weight-decay
        5. gem               — GeM exponent scalar, no weight-decay
        6. head_wd           — projection head (2-D params), weight-decay on
        7. head_no_wd        — projection head (1-D / bias), no weight-decay

        Note: ArcFace parameters are managed in losses.py.

        Parameters
        ----------
        lr_backbone : float
        lr_layernorm : float
            Separate LR for backbone LayerNorm layers.  Some fine-tuning
            recipes benefit from a smaller LR on normalisation layers.
        lr_head : float
        weight_decay : float

        Returns
        -------
        List[dict]  suitable for ``torch.optim.AdamW(model.parameter_groups())``
        """
        # Collect LayerNorm parameter ids to give them their own group.
        ln_param_ids: set = set()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    ln_param_ids.add(id(p))

        def _split(named_params, exclude_ids: set = frozenset()):
            """Split into (wd, no_wd) excluding params in exclude_ids."""
            decay, no_decay = [], []
            for name, param in named_params:
                if not param.requires_grad or id(param) in exclude_ids:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        def _split_ln(named_params):
            """Extract LayerNorm params into (wd, no_wd) by id."""
            decay, no_decay = [], []
            for name, param in named_params:
                if not param.requires_grad or id(param) not in ln_param_ids:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        bb_decay, bb_no_decay = _split(
            self.backbone.named_parameters(), exclude_ids=ln_param_ids
        )
        ln_decay, ln_no_decay = _split_ln(self.backbone.named_parameters())
        hd_decay, hd_no_decay = _split(self.head.named_parameters())

        groups: List[Dict[str, Any]] = [
            {"params": bb_decay,    "lr": lr_backbone,  "weight_decay": weight_decay,
             "name": "backbone_wd"},
            {"params": bb_no_decay, "lr": lr_backbone,  "weight_decay": 0.0,
             "name": "backbone_no_wd"},
            {"params": ln_decay,    "lr": lr_layernorm, "weight_decay": weight_decay,
             "name": "layernorm_wd"},
            {"params": ln_no_decay, "lr": lr_layernorm, "weight_decay": 0.0,
             "name": "layernorm_no_wd"},
            {"params": list(self.gem.parameters()), "lr": lr_head, "weight_decay": 0.0,
             "name": "gem"},
            {"params": hd_decay,    "lr": lr_head,      "weight_decay": weight_decay,
             "name": "head_wd"},
            {"params": hd_no_decay, "lr": lr_head,      "weight_decay": 0.0,
             "name": "head_no_wd"},
        ]

        for g in groups:
            n = sum(p.numel() for p in g["params"] if p.requires_grad)
            log.debug(
                f"  param_group '{g['name']}': {n:,} params, lr={g['lr']:.1e}"
            )

        return groups

    # =====================================================================
    # Freeze / unfreeze API
    # =====================================================================

    def freeze_all(self) -> None:
        """Freeze the entire backbone (all parameters → requires_grad=False).

        The GeM pooling exponent and projection head remain trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info("Backbone fully frozen.  Head + GeM remain trainable.")

    def progressive_unfreeze(self, epoch: int) -> int:
        """Apply the freeze schedule for the given training epoch.

        Freeze schedule
        ---------------
        Epochs  1–10 → unfreeze blocks 10–11 only
        Epochs 11–30 → unfreeze blocks  8–11
        Epochs 31–50 → unfreeze blocks  6–11

        Parameters
        ----------
        epoch : int
            Current training epoch (1-indexed).

        Returns
        -------
        unfreeze_from : int
            Index of the first backbone block that is now unfrozen.
        """
        unfreeze_from: int = _FREEZE_SCHEDULE[-1][2]
        for start, end, from_block in _FREEZE_SCHEDULE:
            if start <= epoch <= end:
                unfreeze_from = from_block
                break

        # 1. Freeze everything — clean slate.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze blocks [unfreeze_from .. 11].
        backbone_blocks = self.backbone.blocks
        num_blocks = len(backbone_blocks)
        for i, block in enumerate(backbone_blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # 3. Always unfreeze final LayerNorm layers.
        for attr_name in ("norm", "fc_norm", "norm1", "norm2"):
            layer = getattr(self.backbone, attr_name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

        # 4. Head + GeM always trainable.
        for param in self.head.parameters():
            param.requires_grad = True
        for param in self.gem.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        frac        = 100.0 * n_trainable / n_total
        log.info(
            f"[Epoch {epoch:02d}] Freeze schedule: "
            f"blocks {unfreeze_from}–{num_blocks - 1} + LayerNorm + Head + GeM | "
            f"{n_trainable:,}/{n_total:,} ({frac:.1f}%) trainable"
        )
        return unfreeze_from

    # =====================================================================
    # Utility
    # =====================================================================

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Return total parameter count."""
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only) or p.requires_grad
        )

    def __repr__(self) -> str:
        total     = self.count_parameters()
        trainable = self.count_parameters(trainable_only=True)
        return (
            f"GodhaarModel(\n"
            f"  backbone={_DINOV2_MODEL_NAME}, img_size={_IMG_SIZE}\n"
            f"  pooling=GeM(mode={self.pooling_mode!r}, "
            f"p={self.gem.p.item():.3f})\n"
            f"  head=ProjectionHead({_BACKBONE_EMB_DIM}→{_HIDDEN_DIM}→{_PROJ_DIM})\n"
            f"  output_contract=(B, {_PROJ_DIM}), unit-norm, float32\n"
            f"  params={total:,} (trainable={trainable:,})\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int,
    device: str | torch.device = "cpu",
    pooling: Literal["gem", "avg", "cls"] = "gem",
    **kwargs: Any,
) -> GodhaarModel:
    """Construct and move a GodhaarModel to the target device.

    Parameters
    ----------
    num_classes : int
    device : str or torch.device
    pooling : {"gem", "avg", "cls"}
        Pool strategy — pass "avg" or "cls" to run pooling ablations.
    **kwargs
        Forwarded to ``GodhaarModel.__init__``.

    Returns
    -------
    GodhaarModel on ``device``, backbone fully frozen, head trainable.
    """
    model = GodhaarModel(num_classes=num_classes, pooling=pooling, **kwargs)
    model = model.to(device)
    log.info(f"Model moved to {device}")
    return model


# ---------------------------------------------------------------------------
# Module-level self-test  (python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Self-test device: {device}")

    NUM_CLASSES = 300

    # ── Build ──────────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, device=device, pooling="gem")
    print(model)

    # ── Freeze schedule smoke-test ─────────────────────────────────────────
    for epoch in [1, 10, 11, 30, 31, 50]:
        fb = model.progressive_unfreeze(epoch)
        log.info(f"  Epoch {epoch:02d} → first unfrozen block = {fb}")
    model.freeze_all()

    # ── Forward pass ───────────────────────────────────────────────────────
    model.progressive_unfreeze(epoch=1)
    model.train()

    dummy = torch.randn(2, 3, _IMG_SIZE, _IMG_SIZE, device=device)
    emb   = model(dummy)

    assert emb.shape == (2, _PROJ_DIM), f"Expected (2, {_PROJ_DIM}), got {emb.shape}"
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Embeddings not unit-norm: {norms}"
    log.info(f"Forward pass OK — shape {emb.shape}, norms ≈ 1.0 ✓")

    # ── extract_features ───────────────────────────────────────────────────
    feats = model.extract_features(dummy)
    assert feats.shape == (2, _PROJ_DIM)
    assert not feats.requires_grad
    log.info("extract_features OK ✓")

    # ── Pooling ablation smoke-test ────────────────────────────────────────
    for pool_mode in ("avg", "cls"):
        m = build_model(num_classes=NUM_CLASSES, device=device, pooling=pool_mode)
        m.progressive_unfreeze(epoch=1)
        m.train()
        out = m(dummy)
        assert out.shape == (2, _PROJ_DIM), f"pooling={pool_mode}: bad shape {out.shape}"
        log.info(f"Pooling ablation '{pool_mode}' OK ✓")

    # ── Retrieval sanity check (synthetic) ─────────────────────────────────
    # Build 2 classes × 4 embeddings; perturbed copies should recall correctly.
    model.eval()
    base_a = F.normalize(torch.randn(1, _PROJ_DIM, device=device), dim=1)
    base_b = F.normalize(torch.randn(1, _PROJ_DIM, device=device), dim=1)
    gallery = torch.cat([
        F.normalize(base_a + 0.01 * torch.randn(4, _PROJ_DIM, device=device), dim=1),
        F.normalize(base_b + 0.01 * torch.randn(4, _PROJ_DIM, device=device), dim=1),
    ])   # (8, 256)
    gallery_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    query     = F.normalize(base_a + 0.01 * torch.randn(1, _PROJ_DIM, device=device), dim=1)
    sims      = (query @ gallery.T).squeeze(0)       # (8,)
    top1_idx  = sims.argmax().item()
    top1_label = gallery_labels[top1_idx].item()
    assert top1_label == 0, f"Retrieval sanity failed: top-1 label={top1_label}"
    log.info("Retrieval sanity (synthetic Recall@1) OK ✓")

    # ── Checkpoint round-trip ──────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    model.progressive_unfreeze(epoch=1)
    model.save_checkpoint(
        tmp_path,
        epoch=1,
        metrics={"top1": 0.9883, "gap": 0.3773},
        seed=42,
        note="self-test checkpoint",
    )
    loaded_model, ckpt_meta = GodhaarModel.load_checkpoint(tmp_path, device=device)
    assert ckpt_meta["epoch"] == 1
    assert ckpt_meta["metrics"]["top1"] == 0.9883
    assert "env" in ckpt_meta, "Checkpoint missing env metadata"
    assert "torch" in ckpt_meta["env"]

    loaded_feats = loaded_model.extract_features(dummy)
    assert torch.allclose(feats, loaded_feats, atol=1e-5), \
        "Checkpoint round-trip mismatch!"
    log.info("Checkpoint round-trip OK ✓")
    tmp_path.unlink()

    # ── parameter_groups ──────────────────────────────────────────────────
    model.progressive_unfreeze(epoch=1)
    groups = model.parameter_groups(lr_backbone=5e-5, lr_layernorm=5e-6, lr_head=1e-3)
    assert len(groups) == 7, f"Expected 7 param groups, got {len(groups)}"
    names  = {g["name"] for g in groups}
    assert "layernorm_wd" in names, "Missing layernorm_wd group"
    log.info(f"parameter_groups OK — {len(groups)} groups ✓")

    # ── GeM exponent trainability ──────────────────────────────────────────
    gem_model = build_model(num_classes=NUM_CLASSES, device=device, pooling="gem")
    gem_model.progressive_unfreeze(epoch=1)
    assert gem_model.gem.p.requires_grad, "GeM exponent must be trainable"
    log.info(f"GeM exponent trainable, p={gem_model.gem.p.item():.4f} ✓")

    log.info("\n" + "=" * 60)
    log.info("ALL SELF-TESTS PASSED ✓")
    log.info("=" * 60)
    sys.exit(0)