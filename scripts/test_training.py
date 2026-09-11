"""Tests for the training pipeline.

The repo had none before this. The first one is the important one: it pins the
optimizer invariant whose absence meant production shipped an untrained
backbone for months (fix 15a9115). The new freeze schedule depends on it
entirely — every phase past the first is a silent no-op without it.

Run:  uv run --frozen python -m pytest scripts/test_training.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from model import _FREEZE_SCHEDULE, build_model
from augment import GaussianNoise, JPEGCompression, build_train_transform, build_val_transform

BACKBONE_PARAMS = 86_579_712


@pytest.fixture(scope="module")
def model():
    return build_model(num_classes=300, device=torch.device("cpu"), pooling="gem")


# ---------------------------------------------------------------------------
# THE optimizer invariant
# ---------------------------------------------------------------------------
def test_optimizer_holds_every_backbone_param_at_construction(model):
    """parameter_groups() must hand AdamW the whole backbone, including the
    parts currently frozen. build_model() leaves the backbone fully frozen, so
    a requires_grad filter here yields ZERO backbone params and every later
    unfreeze becomes a no-op — the exact production bug."""
    opt = torch.optim.AdamW(model.parameter_groups(lr_backbone=5e-5, lr_head=1e-3))
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    held = sum(p.numel() for p in model.backbone.parameters() if id(p) in in_opt)
    assert held == BACKBONE_PARAMS, (
        f"optimizer holds {held:,} of {BACKBONE_PARAMS:,} backbone params. "
        "parameter_groups() is filtering on requires_grad again — this is the "
        "regression that shipped an untrained backbone."
    )


@pytest.mark.parametrize("epoch", [s[0] for s in _FREEZE_SCHEDULE] + [s[1] for s in _FREEZE_SCHEDULE])
def test_no_trainable_param_is_orphaned_at_any_phase(model, epoch):
    """After unfreezing at any phase boundary, every trainable param must
    already be in the optimizer built once at the start."""
    opt = torch.optim.AdamW(model.parameter_groups(lr_backbone=5e-5, lr_head=1e-3))
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    model.progressive_unfreeze(epoch)
    orphaned = [p for p in model.parameters() if p.requires_grad and id(p) not in in_opt]
    assert not orphaned, (
        f"epoch {epoch}: {sum(p.numel() for p in orphaned):,} trainable params "
        "are not in the optimizer and would never be updated"
    )


def test_freeze_schedule_every_phase_opens_something(model):
    """A phase that repeats its predecessor adds capacity in name only — the
    old (36,60,9) did exactly that after (16,35,9)."""
    counts = []
    for start, _end, _blk in _FREEZE_SCHEDULE:
        model.progressive_unfreeze(start)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert counts == sorted(counts), f"trainable counts not monotonic: {counts}"
    assert len(set(counts)) == len(counts), f"a phase opened nothing new: {counts}"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def test_jpeg_compression_changes_tensor_and_stays_in_range():
    torch.manual_seed(0)
    img = torch.rand(3, 128, 128)
    out = JPEGCompression(quality=(40, 41))(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, img), "JPEG compression left the tensor untouched"


def test_gaussian_noise_changes_tensor_and_stays_in_range():
    torch.manual_seed(0)
    img = torch.full((3, 64, 64), 0.5)
    out = GaussianNoise(sigma=(0.04, 0.04))(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, img), "noise left the tensor untouched"
    assert out.std() > 0.01, "noise magnitude far below the requested sigma"


def test_train_transform_has_no_blur_and_no_resized_crop():
    """GaussianBlur destroys the exact band the experiment measures;
    RandomResizedCrop's 0.6-1.5 aspect range is far outside the measured
    [0.78, 1.09] and manufactures distortion that never occurs."""
    names = [type(t).__name__ for t in build_train_transform(518).transforms]
    assert "GaussianBlur" not in names
    assert "RandomResizedCrop" not in names
    assert "RandomHorizontalFlip" not in names, "a mirrored print is a different pattern"
    assert "RandomAffine" in names and "ColorJitter" in names and "RandomErasing" in names
    # the affine must be followed by a centre-crop, otherwise it leaves black
    # corners the model can learn and the val transform never produces
    assert names.index("CenterCrop") == names.index("RandomAffine") + 1, names


def test_train_transform_leaves_no_black_corners():
    """Regression for the artifact the first augmentation grid exposed."""
    torch.manual_seed(0)
    img = Image.new("RGB", (1024, 1024), (128, 128, 128))
    tf = build_train_transform(518)
    for _ in range(12):
        out = tf(img)
        corners = torch.stack([out[:, :8, :8], out[:, :8, -8:], out[:, -8:, :8], out[:, -8:, -8:]])
        # normalised black would sit near -2.1; real grey content sits near 0
        assert corners.min() > -1.5, f"black corner leaked through: min={corners.min():.3f}"


def test_val_transform_is_deterministic():
    names = [type(t).__name__ for t in build_val_transform(518).transforms]
    assert names == ["Resize", "ToTensor", "Normalize"], names


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def test_pk_sampler_shape_and_coverage():
    from sampler import PKClusterSampler
    labels = {f"id{i:03d}": i for i in range(60)}
    clusters = {f"id{i:03d}": i % 5 for i in range(60)}
    idx_by_id = {k: list(range(v * 8, v * 8 + 8)) for k, v in labels.items()}
    s = PKClusterSampler(idx_by_id, clusters, P=8, K=4, same_cluster=6, seed=0, batches=200)
    seen, sizes = set(), []
    for batch in s:
        assert len(batch) == 32
        seen.update(batch)
        ids = [i // 8 for i in batch]
        assert len(set(ids)) == 8, "a batch must hold exactly P=8 distinct identities"
        sizes.append(max(np.bincount([clusters[f"id{i:03d}"] for i in set(ids)])))
    assert len(seen) >= 8 * 60 * 0.9, "sampler never reaches most images"
    assert np.median(sizes) >= 6, f"median same-cluster identities per batch = {np.median(sizes)}, want >= 6"


# ---------------------------------------------------------------------------
# Eval protocol selection
#
# Regression test for the bug that destroyed the first A100 run. The protocol
# was chosen by testing whether the integer label SETS were disjoint, but
# load_folder_corpus numbers train and val independently from 0 -- so on a
# genuinely identity-disjoint split the val labels (0..44) are a strict subset
# of the train labels (0..230), `isdisjoint` returns False, and the evaluator
# silently ran closed-set retrieval, scoring different animals that happened to
# share an integer as genuine matches.
# ---------------------------------------------------------------------------
def test_folder_corpus_label_spaces_overlap_so_labels_cannot_pick_protocol(tmp_path):
    """Pins the precondition that made the bug possible, so that if label
    numbering ever changes, whoever changes it sees this explicitly."""
    from corpus import load_folder_corpus
    from PIL import Image
    for i in range(15):
        d = tmp_path / f"{i:03d}"
        d.mkdir()
        for j in range(4):
            Image.new("RGB", (32, 32), (i * 7 % 256, j * 9 % 256, 0)).save(d / f"{j}.jpg")
    train, val = load_folder_corpus(tmp_path, n_val_identities=3, seed=0, min_images=3)

    assert set(train.identities).isdisjoint(set(val.identities)), \
        "split must be identity-disjoint -- that is the whole point of it"
    assert not set(train.labels).isdisjoint(set(val.labels)), (
        "train and val integer labels overlap by construction; any code that "
        "infers identity-disjointness from integer labels is therefore wrong")


def test_evaluate_accepts_explicit_identity_disjoint_flag():
    """The protocol must be caller-supplied, not inferred from labels."""
    import inspect
    from train_dinov2_arcface import evaluate
    params = inspect.signature(evaluate).parameters
    assert "identity_disjoint" in params, \
        "evaluate() must take an explicit identity_disjoint flag"
    assert params["identity_disjoint"].default is None, \
        "flag should default to None (infer + warn), not to a silent bool"


def test_training_passes_identity_disjoint_from_identity_strings():
    """The call site must derive the flag from identity STRINGS, and pass it."""
    src = (Path(__file__).parent / "train_dinov2_arcface.py").read_text(encoding="utf-8")
    assert "split_is_identity_disjoint = set(train_corpus.identities).isdisjoint(" in src, \
        "protocol must be decided from identity strings, where they are still in scope"
    assert "identity_disjoint=split_is_identity_disjoint" in src, \
        "the computed flag must actually reach evaluate()"
