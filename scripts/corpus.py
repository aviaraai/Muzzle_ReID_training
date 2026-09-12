"""Dataset loading for both training stages, behind one interface.

Two on-disk formats exist and neither is going to change:

  folder  : the 300-identity pretraining corpus -- <identity>/IMG_xxxx.jpg,
            no manifest. Identity is the directory name.
  manifest: the Uttarakhand split -- data/CattleMuzzle/<CattleID>/<Image>
            described by benchmark_manifest.csv with a Split column.

Both produce the same (paths, labels, label_map) triple so the training loop
does not branch on format.

The identity-disjointness assertion is not decoration. The entire week's
measurement work rests on train and eval never sharing an animal, and the one
time that was violated -- 79.4% of the old training set being byte-identical
to the old benchmark -- every number produced was inflated and nobody noticed
for months.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_SPLIT_HASH = "1574f9bc96497e002209c475a9aea631ed89010a63a0cf29521de7cbe8a81dc0"


class Corpus:
    """paths/labels for one split, plus the index structure the PK sampler needs."""

    def __init__(self, paths: list[Path], identities: list[str], label_map: dict[str, int]):
        self.paths = paths
        self.identities = identities
        self.label_map = label_map
        self.labels = [label_map[i] for i in identities]

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def indices_by_identity(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, ident in enumerate(self.identities):
            out.setdefault(ident, []).append(i)
        return out

    def __len__(self) -> int:
        return len(self.paths)


def _assert_disjoint(train: Corpus, val: Corpus) -> None:
    a, b = set(train.identities), set(val.identities)
    overlap = a & b
    if overlap:
        raise SystemExit(
            f"ABORT: train and val share {len(overlap)} identities "
            f"(e.g. {sorted(overlap)[:5]}). The split is not identity-disjoint, so "
            "every metric it produces would be inflated. Refusing to train."
        )


def load_folder_corpus(root: Path, n_val_identities: int = 45, seed: int = 20260911,
                       min_images: int = 3) -> tuple[Corpus, Corpus]:
    """The 300-identity pretraining corpus. Identity == directory name."""
    root = Path(root)
    by_id: dict[str, list[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(imgs) >= min_images:
            by_id[d.name] = imgs
    if not by_id:
        raise SystemExit(f"ABORT: no identities with >= {min_images} images under {root}")

    ids = sorted(by_id)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = min(n_val_identities, max(1, len(ids) // 5))
    val_ids, train_ids = set(shuffled[:n_val]), set(shuffled[n_val:])

    # labels are assigned over TRAIN identities only -- ArcFace has no class
    # for a held-out animal, which is the point of an open-set protocol
    label_map = {i: n for n, i in enumerate(sorted(train_ids))}
    val_map = {i: n for n, i in enumerate(sorted(val_ids))}

    def build(idset: set[str], lm: dict[str, int]) -> Corpus:
        paths, idents = [], []
        for i in sorted(idset):
            for p in by_id[i]:
                paths.append(p)
                idents.append(i)
        return Corpus(paths, idents, lm)

    train, val = build(train_ids, label_map), build(val_ids, val_map)
    _assert_disjoint(train, val)
    return train, val


def load_manifest_corpus(manifest: Path, data_root: Path,
                         require_split_hash: str | None = None) -> tuple[Corpus, Corpus]:
    """The Uttarakhand split: benchmark_manifest.csv + data/CattleMuzzle/."""
    manifest, data_root = Path(manifest), Path(data_root)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"ABORT: {manifest} is empty")

    if require_split_hash:
        found = _manifest_split_hash(manifest)
        if found != require_split_hash:
            raise SystemExit(
                f"ABORT: stage-2 requires split_hash {require_split_hash}, "
                f"but this manifest reports {found!r}. Refusing to fine-tune on a "
                "split that is not the one the benchmark was locked against."
            )

    def build(split: str) -> Corpus:
        sel = [r for r in rows if r["Split"] == split]
        idents = [r["CattleID"] for r in sel]
        lm = {i: n for n, i in enumerate(sorted(set(idents)))}
        return Corpus([data_root / r["CattleID"] / r["Image"] for r in sel], idents, lm)

    train, val = build("gallery"), build("query")
    _assert_disjoint(train, val)
    return train, val


def _manifest_split_hash(manifest: Path) -> str | None:
    """The split hash lives in splits/split_v1.json, not in the CSV. Look for
    it next to the manifest and one level up, so stage-2 can verify it is
    fine-tuning against the locked split rather than some other manifest that
    happens to have the right column names."""
    for cand in (manifest.parent / "split_v1.json",
                 manifest.parent.parent / "splits" / "split_v1.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8")).get("split_hash")
            except Exception:
                return None
    return None


def load_clusters(path: Path, identities: set[str]) -> dict[str, int]:
    """Appearance-cluster assignment for the PK sampler. Missing file or
    missing identities degrade to one cluster per identity, which makes the
    sampler behave like plain random P x K rather than failing the run."""
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found -- PK sampler falls back to random identity draw")
        return {i: n for n, i in enumerate(sorted(identities))}
    data = json.loads(path.read_text(encoding="utf-8")).get("assignment", {})
    missing = identities - set(data)
    if missing:
        print(f"  WARNING: {len(missing)} identities absent from cluster file; "
              "assigning each its own cluster")
    out = dict(data)
    for n, i in enumerate(sorted(missing)):
        out[i] = 10_000 + n
    return {i: out[i] for i in identities}


REQUIRED_SPLIT_300_HASH = "2cffb3787eae89e1b57d21a82d7808d9a0d76c380ad8e9005e121bc779f851ef"


def load_split_corpus(split_path: Path, corpus_dir: Path,
                      require_hash: str | None = REQUIRED_SPLIT_300_HASH
                      ) -> tuple[Corpus, Corpus, dict]:
    """The 300-corpus as a LOCKED split manifest, not a live folder scan.

    load_folder_corpus() re-derives the split from whatever happens to be on
    disk every run, so a corpus that gains, loses or re-crops a file silently
    changes what "val" means and two runs stop being comparable. This reads a
    fixed manifest instead: deduplicated, identity-reconciled, identity-disjoint,
    and hashed, so a run either scores the instrument it claims to or refuses to
    start.

    Paths in the manifest are RELATIVE and resolved against `corpus_dir`, so the
    same file works on the machine that built it and the server that trains on
    it. Every image's sha256 is verified -- a silently re-encoded or replaced
    crop is caught here rather than quietly shifting a number later.

    Returns (train, val, split_dict); the caller needs the dict for
    `eval_protocol`.
    """
    split_path, corpus_dir = Path(split_path), Path(corpus_dir)
    if not split_path.exists():
        raise SystemExit(f"ABORT: split manifest not found: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if require_hash and split.get("split_hash") != require_hash:
        raise SystemExit(
            f"ABORT: {split_path.name} has split_hash {split.get('split_hash')!r}, "
            f"expected {require_hash!r}. A different split is a different "
            f"instrument and its numbers are not comparable to any recorded "
            f"baseline. Refusing to train."
        )

    import hashlib

    # Both sections are checked BEFORE anything aborts. Raising inside the
    # train pass meant val was never examined, so a corpus with damage in both
    # halves surfaced one file per run -- fix, re-copy, re-run, discover the
    # next one. Whoever is repairing a transfer needs the whole list at once.
    def build(section: str) -> tuple[Corpus, list[str]]:
        ids = sorted(split[section])
        lm = {i: n for n, i in enumerate(ids)}
        paths, idents, bad = [], [], []
        for i in ids:
            for e in split[section][i]:
                p = corpus_dir / e["rel"]
                if not p.exists():
                    bad.append(f"{section}/{e['rel']}: missing")
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != e["sha256"]:
                    bad.append(f"{section}/{e['rel']}: sha256 mismatch "
                               f"({p.stat().st_size} bytes on disk)")
                    continue
                paths.append(p)
                idents.append(i)
        return Corpus(paths, idents, lm), bad

    train, bad_train = build("train")
    val, bad_val = build("val")
    bad = bad_train + bad_val
    if bad:
        listed = "\n  ".join(bad[:20])
        more = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
        raise SystemExit(
            f"ABORT: {len(bad)} of {len(train) + len(val) + len(bad)} image(s) are missing "
            f"or altered under {corpus_dir}:\n  {listed}{more}\n"
            f"The split describes an instrument that is no longer on disk, so any score "
            f"would be meaningless. Re-copy the listed file(s) from the source corpus -- "
            f"a mismatch is almost always a truncated or interrupted transfer, not a bad "
            f"split. Run scripts/verify_corpus.py for the full report without starting a run."
        )
    _assert_disjoint(train, val)
    return train, val, split
