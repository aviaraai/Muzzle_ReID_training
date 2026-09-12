"""Verify a transferred corpus against the split manifest, without training.

A split manifest records the sha256 of every image it describes, so a corpus
that arrived over a pendrive, rsync or an unzip can be checked exactly rather
than by eyeballing file counts. A truncated or interrupted copy produces a file
of the right name and roughly the right size that hashes differently -- which is
invisible until a run aborts on it.

Reports EVERY problem in one pass, both sections, so a damaged transfer is
repaired once instead of one file per failed run. Read-only.

    uv run --frozen python scripts/verify_corpus.py \
        --corpus-dir /path/to/300-crops-arm2 \
        --split uk_benchmark/splits/split_300_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--split", default=str(ROOT / "uk_benchmark" / "splits" / "split_300_v1.json"))
    a = ap.parse_args()

    corpus, split_path = Path(a.corpus_dir), Path(a.split)
    if not split_path.exists():
        print(f"FAIL: split manifest not found: {split_path}")
        return 2
    if not corpus.is_dir():
        print(f"FAIL: corpus directory not found: {corpus}")
        return 2

    split = json.loads(split_path.read_text(encoding="utf-8"))
    print(f"split      : {split_path.name}  hash {split.get('split_hash', '?')[:8]}")
    print(f"corpus     : {corpus}")

    marker = corpus / "_ARM2_CROP_MARKER.json"
    print(f"crop marker: {'present' if marker.exists() else 'MISSING -- --arm crop will abort'}")

    missing, altered, ok = [], [], 0
    for section in ("train", "val"):
        for ident in sorted(split[section]):
            for e in split[section][ident]:
                p = corpus / e["rel"]
                if not p.exists():
                    missing.append(e["rel"])
                elif sha256(p) != e["sha256"]:
                    altered.append((e["rel"], p.stat().st_size))
                else:
                    ok += 1

    total = ok + len(missing) + len(altered)
    print(f"\nchecked    : {total} images")
    print(f"  intact   : {ok}")
    print(f"  missing  : {len(missing)}")
    print(f"  altered  : {len(altered)}")

    for r in missing[:40]:
        print(f"    MISSING  {r}")
    for r, sz in altered[:40]:
        print(f"    ALTERED  {r}  ({sz} bytes on disk)")
    if len(missing) + len(altered) > 80:
        print(f"    ... {len(missing) + len(altered) - 80} more")

    if missing or altered:
        print(f"\nRe-copy the file(s) above from the source corpus. A mismatch is almost "
              f"always a truncated or interrupted transfer, not a bad split -- the "
              f"manifest's hashes were verified against the source when it was built.")
        return 1
    print("\nOK: every image matches the manifest. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
