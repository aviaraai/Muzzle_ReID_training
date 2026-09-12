"""eval/benchmark.py — the single scoring instrument.

Every experiment calls evaluate() and nothing computes its own metrics. That
is the whole point: numbers from two different experiments are only
comparable if they came from the same split and the same aggregation, and the
only way to guarantee that is to have one implementation.

    from eval.benchmark import evaluate
    result = evaluate(model.embed_images, experiment_name="fusion_v1")

`embed_fn` takes a list of raw image BYTES and returns an (N, D) array — the
same contract as pipeline/fusion_encoder.py::embed_images and
pipeline/muzzle.py::embed_batch, so a production encoder can be passed
directly with no adapter.

Protocol
--------
Leave-one-out over the benchmark split only. Each benchmark image is a query;
the gallery is every OTHER benchmark image, including the query animal's own
remaining images. Scores are aggregated per animal by MAX, animals are
ranked, and top-k asks whether the query's true animal is in the top k.

The aggregation deliberately mirrors go-apiserver, so an offline number here
means the same thing as an online decision there:

  * per-animal max, strict `>` so the first-seen embedding wins an exact tie
    — go-apiserver internal/server/mobile/handlers/animal/routes.go:498
  * rank ties broken by animal id ascending
    — internal/server/mobile/handlers/animal/decision.go:214-217

Deviating from either would make these metrics quietly incomparable to
production behaviour, so both are asserted against the split, not assumed.

One asymmetry to keep in mind when reading `separation`
-------------------------------------------------------
Leave-one-out removes the query from the gallery, so the query's OWN animal
is represented by k-1 images while every impostor animal keeps all k. Since
the aggregation is a max, and the max of 3 draws exceeds the max of 2 in
expectation, this handicaps the genuine score by construction. Measured with
random 64-d Gaussian embeddings on this split: top1 = 0.0000 and separation =
-0.2328, where a naive reading would predict 0. That is the floor of the
protocol, not a defect, and it means:

  * separation ~= 0 is already meaningfully better than chance here
  * a NEGATIVE separation is not automatically damning -- compare it against
    the -0.23 random floor, not against 0
  * production is slightly kinder than this benchmark: there a query is a
    fresh photo and the enrolled animal keeps all k of its embeddings, so
    LOO understates real-world separation rather than flattering it

Every result file carries `split_hash`. A result whose split_hash differs
from another's was measured on a different instrument and the two must not be
compared.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = ["evaluate", "load_split"]

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_SPLIT = _REPO / "splits" / "split_v1.json"
_RESULTS_DIR = _REPO / "results"
_PROTECTED_DIR = _REPO / "benchmark_images"


def load_split(split_path: str | Path = _DEFAULT_SPLIT) -> dict:
    """Read a split manifest. Kept public so an experiment can inspect the
    split without duplicating the path logic."""
    return json.loads(Path(split_path).read_text(encoding="utf-8"))


def _resolve(entry: dict, canonical_id: str) -> Path:
    """Prefer the read-only benchmark copy over the mutable source tree.

    benchmark_images/ is chmod'd read-only precisely so the scored bytes
    cannot drift; the manifest's `path` points at the original corpus, which
    is not protected. Either way the bytes are verified against the recorded
    sha256 below, so a silently-edited file is caught rather than scored.
    """
    protected = _PROTECTED_DIR / canonical_id / Path(entry["path"]).name
    return protected if protected.exists() else Path(entry["path"])


def _load_benchmark_images(split: dict) -> tuple[list[bytes], list[str], list[str]]:
    blobs: list[bytes] = []
    animals: list[str] = []
    paths: list[str] = []
    mismatches: list[str] = []

    for cid in sorted(split["benchmark"]):
        for entry in split["benchmark"][cid]:
            p = _resolve(entry, cid)
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                mismatches.append(str(p))
                continue
            blobs.append(raw)
            animals.append(cid)
            paths.append(str(p))

    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} benchmark image(s) do not match the sha256 recorded in the "
            f"split manifest — the instrument has been altered and any score would be "
            f"meaningless. Restore them before evaluating:\n  "
            + "\n  ".join(mismatches[:10])
        )
    return blobs, animals, paths


def _embed_all(embed_fn: Callable[[list[bytes]], Sequence], blobs: list[bytes], batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(blobs), batch_size):
        e = embed_fn(blobs[i:i + batch_size])
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        out.append(np.asarray(e, dtype=np.float64))
    embs = np.vstack(out)
    if embs.shape[0] != len(blobs):
        raise RuntimeError(f"embed_fn returned {embs.shape[0]} vectors for {len(blobs)} images")
    # Defensive re-normalisation: cosine similarity is only a dot product if
    # the vectors are unit-length, and not every encoder guarantees it.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise RuntimeError("embed_fn produced a zero or non-finite embedding")
    return embs / norms


def evaluate(
    embed_fn: Callable[[list[bytes]], Sequence],
    experiment_name: str | None = None,
    split_path: str | Path = _DEFAULT_SPLIT,
    batch_size: int = 16,
    write_result: bool = True,
) -> dict:
    """Score `embed_fn` on the benchmark split. Returns the metrics dict and
    (unless disabled) writes results/<experiment_name>_<split_hash[:8]>.json.
    """
    if experiment_name is None:
        experiment_name = getattr(embed_fn, "__qualname__", None) or getattr(embed_fn, "__name__", "unnamed")
        experiment_name = experiment_name.replace(".", "_").replace("<", "").replace(">", "")

    split = load_split(split_path)
    split_hash = split["split_hash"]

    blobs, animals, paths = _load_benchmark_images(split)
    n = len(blobs)
    uniq = sorted(set(animals))
    if n == 0:
        raise RuntimeError("benchmark split is empty")

    embs = _embed_all(embed_fn, blobs, batch_size)
    sim = embs @ embs.T
    np.fill_diagonal(sim, -np.inf)          # leave-one-out: a query is never its own gallery entry

    animals_arr = np.asarray(animals)
    # Column blocks per animal, in the same ascending-id order go-apiserver
    # breaks ties on, so argmax picks the identical winner on an exact tie.
    order = {cid: i for i, cid in enumerate(uniq)}
    col_idx = [np.where(animals_arr == cid)[0] for cid in uniq]

    top1 = top5 = top10 = 0
    genuine, impostor = [], []

    for q in range(n):
        # per-animal MAX over the gallery (strict >, first-seen wins ties ->
        # np.max over an ascending-id-ordered block is equivalent)
        agg = np.array([sim[q, idx].max() for idx in col_idx])

        true_col = order[animals[q]]
        genuine.append(agg[true_col])
        impostor.extend(np.delete(agg, true_col).tolist())

        # rank: score desc, ties broken by animal id ascending. lexsort's last
        # key is primary, so this sorts by (-score, id-index).
        ranking = np.lexsort((np.arange(len(uniq)), -agg))
        rank = int(np.where(ranking == true_col)[0][0])
        top1 += rank < 1
        top5 += rank < 5
        top10 += rank < 10

    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    pooled_sd = np.sqrt(0.5 * (genuine.var(ddof=1) + impostor.var(ddof=1)))
    dprime = float((genuine.mean() - impostor.mean()) / pooled_sd) if pooled_sd > 0 else float("inf")

    result = {
        "experiment_name": experiment_name,
        "split_hash": split_hash,
        "split_version": split["split_version"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_queries": n,
        "n_animals": len(uniq),
        "top1": top1 / n,
        "top5": top5 / n,
        "top10": top10 / n,
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "separation": float(genuine.mean() - impostor.mean()),
        "dprime": dprime,
        "embedding_dim": int(embs.shape[1]),
    }

    if write_result:
        _RESULTS_DIR.mkdir(exist_ok=True)
        out = _RESULTS_DIR / f"{experiment_name}_{split_hash[:8]}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["result_path"] = str(out)

    return result
