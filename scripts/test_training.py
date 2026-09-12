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


# ---------------------------------------------------------------------------
# The two instruments
#
# In-corpus val drives early stopping; the cross-population benchmark is
# read-only. These pin the properties that make the second one meaningful --
# if the benchmark ever reaches a checkpoint-selection or early-stopping
# decision, it stops measuring transfer and starts measuring how hard the run
# was fitted to it, and no later number from it can be trusted.
# ---------------------------------------------------------------------------
def test_images_per_identity_changes_the_score_monotonically():
    """The pinned protocol exists because this variable dominates the metric."""
    from instruments import score_fixed_shots
    # Noise/dim chosen so the task is genuinely hard (Top-1 ~0.44 at k=2). At
    # low noise every setting scores 1.000 and the test proves nothing.
    rng = np.random.default_rng(0)
    n_ids, per_id, dim = 30, 6, 16
    centres = rng.normal(size=(n_ids, dim))
    embs, labels = [], []
    for i in range(n_ids):
        v = centres[i] + 1.0 * rng.normal(size=(per_id, dim))
        embs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        labels.extend([i] * per_id)
    embs = np.vstack(embs).astype(np.float32)
    labels = np.array(labels)

    scores = [score_fixed_shots(embs, labels, k, draws=5, seed=1)["top1"]
              for k in (2, 3, 4, 5)]
    assert scores == sorted(scores), (
        f"Top-1 must not fall as shots per identity rise; got {scores}")
    assert scores[-1] > scores[0], "more shots per identity must make LOO easier"


def test_benchmark_is_never_used_for_selection_or_early_stopping():
    """Static proof: the benchmark's result must not reach any decision var."""
    src = (Path(__file__).parent / "train_dinov2_arcface.py").read_text(encoding="utf-8")
    # Scope strictly to the benchmark block: from the call to the end of its
    # own except clause. A wider window would sweep in later, unrelated code
    # that legitimately mentions best_top1 and make this pass or fail for the
    # wrong reason.
    body = src[src.index("if run_benchmark:"):]
    start = body.index("score_benchmark(")
    end = body.index("cross-population scoring failed")
    window = body[start:end]
    for forbidden in ("best_top1", "best_gap", "epochs_no_improve", "is_best",
                      "save_checkpoint", "metrics["):
        assert forbidden not in window, (
            f"cross-population scoring must not touch {forbidden!r} -- a read-only "
            f"instrument that influences training measures nothing")
    assert "benchmark_history.append" in window, "the score must still be recorded"


def test_benchmark_scored_only_at_phase_boundaries():
    from train_dinov2_arcface import _PHASE_BOUNDARY_EPOCHS
    assert _PHASE_BOUNDARY_EPOCHS == {5, 20, 45, 80}, (
        f"benchmark epochs are the freeze-phase boundaries; got {_PHASE_BOUNDARY_EPOCHS}")


def test_leakage_assertion_catches_an_exact_and_a_near_duplicate(tmp_path):
    """It must fail on both byte-identical and merely re-encoded overlap."""
    from instruments import assert_disjoint_from_benchmark
    from PIL import Image
    bench = tmp_path / "bench"; bench.mkdir()
    train = tmp_path / "train"; train.mkdir()
    rng = np.random.default_rng(3)
    shared = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    shared.save(bench / "b0.jpg", quality=95)
    for i in range(3):
        a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        a.save(train / f"t{i}.jpg", quality=95)
    # clean -> passes
    ok = assert_disjoint_from_benchmark(sorted(train.glob("*.jpg")), bench)
    assert ok["exact_collisions"] == 0 and ok["phash_collisions"] == 0

    # exact copy -> must abort
    import shutil
    shutil.copy(bench / "b0.jpg", train / "leak.jpg")
    with pytest.raises(SystemExit, match="overlaps the benchmark"):
        assert_disjoint_from_benchmark(sorted(train.glob("*.jpg")), bench)

    # re-encoded copy (different sha256, same photograph) -> must still abort
    (train / "leak.jpg").unlink()
    Image.open(bench / "b0.jpg").save(train / "reenc.jpg", quality=60)
    assert _sha(train / "reenc.jpg") != _sha(bench / "b0.jpg"), "test setup: sha must differ"
    with pytest.raises(SystemExit, match="overlaps the benchmark"):
        assert_disjoint_from_benchmark(sorted(train.glob("*.jpg")), bench)


def _sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_vendored_benchmark_resolves_its_own_images_and_refuses_to_fall_back():
    """The vendored harness sits one level shallower than the original.

    parent.parent pointed _PROTECTED_DIR at a directory that does not exist, so
    _resolve() fell back to the manifest's ABSOLUTE source path -- which exists
    only on the machine that built the split. The instrument therefore worked
    locally and failed on the training server, silently, for a whole 80-epoch
    run. Both halves are pinned: the directory must resolve, and the fallback
    must be gone.
    """
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from uk_benchmark import benchmark as B

    assert B._REPO.name == "uk_benchmark", (
        f"_REPO must be the uk_benchmark dir itself, got {B._REPO}")
    assert B._DEFAULT_SPLIT.exists(), f"split manifest missing: {B._DEFAULT_SPLIT}"

    src = (root / "uk_benchmark" / "benchmark.py").read_text(encoding="utf-8")
    assert "return protected if protected.exists() else Path(entry[" not in src, (
        "the silent absolute-path fallback must stay removed -- it is what hid "
        "a missing benchmark_images/ through a full training run")
    assert "raise FileNotFoundError" in src, "a missing image must fail loudly"
