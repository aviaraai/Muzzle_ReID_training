"""The two measuring instruments a pretraining run is scored against.

They answer different questions and must never be conflated:

  IN-CORPUS VAL   45 identities held out of the 300-corpus itself
                  (split_300_v1.json). Same imagery, same camera, same
                  conditions as training -- so it measures "did the model learn
                  these animals' muzzles" and nothing about transfer. Cheap
                  enough to run every epoch, and it is what early stopping
                  watches.

  CROSS-POPULATION  the locked Uttarakhand benchmark (split_v1.json,
                    split_hash 1574f9bc...). Different cattle, different
                    photographer, different conditions -- production's actual
                    distribution. This is the number that says whether
                    pretraining transferred.

The benchmark is READ-ONLY with respect to training: it is scored at phase
boundaries only, never drives checkpoint selection or early stopping, and
nothing in this module writes it back into any training decision. The moment a
held-out instrument influences the run, it stops measuring generalisation and
starts measuring how hard the run was fitted to it.

Both are scored under the SAME retrieval protocol -- leave-one-out,
per-identity MAX aggregation, ranked -- with images-per-identity pinned, because
that single variable moved stock DINOv2 from 0.6956 to 0.9484 on identical
imagery and is the difference between an instrument with headroom and one that
reads saturated.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("godhaar.instruments")

ROOT = Path(__file__).resolve().parent.parent
PHASH_THRESHOLD = 4


# ── hashing ──────────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash64(path_or_bytes) -> np.uint64:
    """Difference hash, 9x8 grayscale, horizontally adjacent comparisons.

    `draft` lets libjpeg do the downscale during decode instead of decoding a
    4000x6000 original in full and throwing it away -- this runs over thousands
    of images at startup and the difference is minutes, not seconds.
    """
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    im = Image.open(src)
    try:
        im.draft("L", (9, 8))
    except Exception:
        pass
    im = im.convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = np.uint64(0)
    for b in bits:
        out = np.uint64(out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def _hamming(a: np.ndarray, b: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(a, b)
    c = np.zeros_like(x, dtype=np.uint8)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


# ── leakage ──────────────────────────────────────────────────────────────────
def assert_disjoint_from_benchmark(train_paths: list[Path], benchmark_dir: Path) -> dict:
    """Refuse to start if any training image is also a benchmark image.

    Verified once by hand (0 exact, 0 phash<=4, closest Hamming distance 12) --
    re-asserted every run so it cannot silently break when the corpus is
    regenerated, re-cropped, or a folder is copied into the wrong place. A
    training set that overlaps the benchmark makes every cross-population number
    meaningless in the flattering direction, which is the failure mode nobody
    notices.

    Checks BOTH exact bytes and perceptual similarity: a re-encoded or re-cropped
    copy of a benchmark photo has a different sha256 but is still the same
    photograph, and would leak just as badly.
    """
    bench = sorted(p for p in benchmark_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not bench:
        raise SystemExit(f"ABORT: no benchmark images under {benchmark_dir}")

    b_sha = {_sha256(p) for p in bench}
    t_sha = [_sha256(p) for p in train_paths]
    exact = sum(1 for h in t_sha if h in b_sha)

    b_hash = np.array([_dhash64(p) for p in bench], dtype=np.uint64)
    t_hash = np.array([_dhash64(p) for p in train_paths], dtype=np.uint64)
    closest = 64
    near = 0
    for h in t_hash:
        d = _hamming(b_hash, h)
        m = int(d.min())
        closest = min(closest, m)
        if m <= PHASH_THRESHOLD:
            near += 1

    result = {"train_images": len(train_paths), "benchmark_images": len(bench),
              "exact_collisions": exact, "phash_collisions": near,
              "closest_hamming": closest}
    if exact or near:
        raise SystemExit(
            f"ABORT: training corpus overlaps the benchmark -- {exact} byte-identical "
            f"and {near} perceptually near-identical (Hamming <= {PHASH_THRESHOLD}) "
            f"image(s), closest distance {closest}. The cross-population number would "
            f"be measuring memorisation. Refusing to train."
        )
    return result


# ── retrieval protocol ───────────────────────────────────────────────────────
def loo_rank_metrics(embs: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-out retrieval, per-identity MAX aggregation, ranked.

    Mirrors go-apiserver's own cattleScores aggregation (max over an animal's
    images, strict >), so a number here means the same thing it would in
    production rather than something only this script computes.
    """
    sim = embs @ embs.T
    n = len(labels)
    ranks, gen, imp = [], [], []
    for i in range(n):
        per_id: dict[int, float] = {}
        for j in range(n):
            if j == i:
                continue
            s = float(sim[i, j])
            l = int(labels[j])
            if l not in per_id or s > per_id[l]:
                per_id[l] = s
        own = int(labels[i])
        if own not in per_id:
            continue
        order = sorted(per_id.items(), key=lambda kv: -kv[1])
        ranks.append([k for k, _ in order].index(own) + 1)
        gen.append(per_id[own])
        others = [s for k, s in per_id.items() if k != own]
        if others:
            imp.append(max(others))
    r = np.array(ranks)
    if len(r) == 0:
        return {"top1": 0.0, "top5": 0.0, "gen": float("nan"),
                "imp": float("nan"), "gap": float("nan"), "n": 0}
    return {"top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "gen": float(np.mean(gen)), "imp": float(np.mean(imp)),
            "gap": float(np.mean(gen) - np.mean(imp)), "n": len(r)}


def score_fixed_shots(embs: np.ndarray, labels: np.ndarray,
                      images_per_identity: int, draws: int, seed: int) -> dict:
    """Score with images-per-identity PINNED, averaged over seeded draws.

    Scoring with every image an identity happens to own makes the metric a
    function of how many photos the field team took of that animal: on identical
    imagery and encoder, Top-1 runs 0.6956 / 0.8304 / 0.8875 / 0.9220 / 0.9484 at
    2 / 3 / 4 / 5 / all. Pinning it is what makes two runs -- and the two
    instruments -- comparable at all. A single draw carries sd ~0.025, so
    averaging several keeps sampling noise from being read as progress.
    """
    by_id: dict[int, np.ndarray] = {}
    for l in np.unique(labels):
        by_id[int(l)] = np.nonzero(labels == l)[0]

    acc: list[dict] = []
    for d in range(draws):
        rng = np.random.default_rng(seed + d)
        sel: list[int] = []
        for l, idxs in by_id.items():
            if len(idxs) >= images_per_identity:
                sel.extend(rng.choice(idxs, images_per_identity, replace=False).tolist())
        if not sel:
            continue
        sel = np.array(sorted(sel))
        acc.append(loo_rank_metrics(embs[sel], labels[sel]))
    if not acc:
        return {"top1": 0.0, "top5": 0.0, "gap": float("nan"), "n": 0, "draws": 0}
    out = {k: float(np.mean([a[k] for a in acc])) for k in ("top1", "top5", "gen", "imp", "gap")}
    out["top1_sd"] = float(np.std([a["top1"] for a in acc]))
    out["gap_sd"] = float(np.std([a["gap"] for a in acc]))
    out["n"] = int(np.mean([a["n"] for a in acc]))
    out["draws"] = len(acc)
    return out


# ── cross-population benchmark ───────────────────────────────────────────────
def score_benchmark(model, device, transform, batch_size: int = 16) -> dict:
    """Score the locked Uttarakhand benchmark through its own harness.

    Calls the vendored eval harness rather than reimplementing its protocol, so
    a number produced mid-training is directly comparable to the recorded
    baselines (production 0.8381 nocrop / 0.7095 crop, stock DINOv2 0.8619).
    `write_result=False`: a training run must not leave artefacts in the
    instrument's own results directory.
    """
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from uk_benchmark.benchmark import evaluate as bench_evaluate

    was_training = model.training
    model.eval()

    @torch.no_grad()
    def embed_fn(blobs: list[bytes]):
        x = torch.stack([transform(Image.open(io.BytesIO(b)).convert("RGB"))
                         for b in blobs]).to(device)
        e = model(x).float().cpu().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

    try:
        res = bench_evaluate(embed_fn, experiment_name="pretrain_inflight",
                             batch_size=batch_size, write_result=False)
    finally:
        if was_training:
            model.train()
    return res
