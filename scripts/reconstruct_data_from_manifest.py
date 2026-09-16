"""reconstruct_data_from_manifest.py — rebuild data/CattleMuzzle/<ID>/<Image>
from an existing local copy of the raw Godhaar_aron folder, using only the
manifest CSV (no image bytes need to be re-transferred if you already have
a copy of the source images on this machine).

Verifies every file it copies against the manifest's own sha256 column, so
another machine's copy of Godhaar_aron that differs even slightly from the one
the manifest was built against (re-compressed, partially re-synced, wrong
version) is caught explicitly -- not silently used as if it were identical.
The quality/crop-gate decisions that produced this manifest were made
against SPECIFIC bytes; a different-but-similar-looking file is not the
same input and must not be swapped in quietly.

Usage:
    python scripts/reconstruct_data_from_manifest.py --source-dir /path/to/Godhaar_aron

Exits non-zero (and lists every problem) if anything is missing or a hash
doesn't match -- fix those before training, don't proceed past a partial or
mismatched reconstruction.
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "benchmark_manifest.csv"
DEST_DIR = ROOT / "data" / "CattleMuzzle"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="Local folder containing the raw Godhaar_aron *.jpg files (flat, not yet split by identity)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"ERROR: --source-dir does not exist or is not a directory: {source_dir}", file=sys.stderr)
        sys.exit(1)

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST} -- did you copy data/benchmark_manifest.csv into place?", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    print(f"{len(rows)} images listed in {MANIFEST}")

    missing, mismatched, copied, already_ok = [], [], 0, 0

    for row in rows:
        src = source_dir / row["Image"]
        dst = DEST_DIR / row["CattleID"] / row["Image"]

        if dst.exists():
            if sha256_of(dst) == row["sha256"]:
                already_ok += 1
                continue
            # dst exists but wrong content -- fall through and re-copy from source

        if not src.exists():
            missing.append(row["Image"])
            continue

        actual_hash = sha256_of(src)
        if actual_hash != row["sha256"]:
            mismatched.append((row["Image"], row["sha256"], actual_hash))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"\n{already_ok} already in place and verified, {copied} copied and verified")

    if missing:
        print(f"\n{len(missing)} file(s) NOT FOUND in --source-dir:")
        for name in missing:
            print(f"  {name}")

    if mismatched:
        print(f"\n{len(mismatched)} file(s) FOUND but content does NOT match "
              f"(this source folder is not byte-identical to the one the "
              f"manifest was built from -- do not proceed with these):")
        for name, expected, actual in mismatched:
            print(f"  {name}: expected sha256={expected[:16]}... got {actual[:16]}...")

    if missing or mismatched:
        print(f"\nFAILED: {len(missing) + len(mismatched)} of {len(rows)} images could not be "
              f"reconstructed correctly. Fix --source-dir (get the missing/correct files from "
              f"wherever the original Godhaar_aron copy lives) before training on this data.")
        sys.exit(1)

    print(f"\nOK: all {len(rows)} images reconstructed and verified at {DEST_DIR}")


if __name__ == "__main__":
    main()
