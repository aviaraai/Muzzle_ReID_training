"""Score saved checkpoints against the cross-population benchmark.

The benchmark is meant to run inside training at phase boundaries, but that call
is wrapped in a try/except so a benchmark failure can never take a run down --
which means a run can complete with no transfer number at all. It did: a path
bug in the vendored harness made every in-flight scoring fail, and the 80-epoch
run finished with an excellent in-corpus curve and nothing to say about whether
any of it transferred.

The phase checkpoints are exactly what makes that recoverable. This scores them
after the fact, which is equivalent to the in-flight measurement -- the
benchmark never influenced training, so scoring it now versus then changes
nothing about what is being measured.

    uv run --frozen python scripts/score_checkpoints.py
    uv run --frozen python scripts/score_checkpoints.py --checkpoints checkpoints/*.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from augment import build_val_transform      # noqa: E402
from config import IMG_SIZE                  # noqa: E402
from model import GodhaarModel               # noqa: E402
from instruments import score_benchmark      # noqa: E402

REFERENCES = {
    "stock DINOv2 (no fine-tuning)": 0.8619,
    "production encoder, nocrop": 0.8381,
    "production encoder, crop": 0.7095,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoint files (default: every .pt under checkpoints/)")
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark_history.json"))
    a = ap.parse_args()

    ckpts = ([Path(c) for c in a.checkpoints] if a.checkpoints
             else sorted((ROOT / "checkpoints").glob("*.pt")))
    ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        print("no checkpoints found")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_val_transform(IMG_SIZE)
    print(f"device: {device}   checkpoints: {len(ckpts)}\n")

    rows = []
    for c in ckpts:
        try:
            model, ck = GodhaarModel.load_checkpoint(c, device=device)
        except Exception as exc:
            print(f"  {c.name:<22} SKIPPED ({exc})")
            continue
        res = score_benchmark(model, device, transform)
        rows.append({"checkpoint": c.name, "epoch": ck.get("epoch"),
                     "top1": res["top1"], "top5": res["top5"],
                     "separation": res["separation"],
                     "in_corpus_top1": (ck.get("metrics") or {}).get("top1"),
                     "in_corpus_gap": (ck.get("metrics") or {}).get("gap")})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"]))
    print(f"{'checkpoint':<22} {'ep':>4} {'IN-CORPUS':>10} {'gap':>8} | "
          f"{'BENCH top1':>11} {'top5':>7} {'sep':>8}")
    print("-" * 80)
    for r in rows:
        ic = f"{r['in_corpus_top1']:.4f}" if r["in_corpus_top1"] is not None else "    -   "
        ig = f"{r['in_corpus_gap']:+.4f}" if r["in_corpus_gap"] is not None else "    -   "
        print(f"{r['checkpoint']:<22} {str(r['epoch']):>4} {ic:>10} {ig:>8} | "
              f"{r['top1']:>11.4f} {r['top5']:>7.4f} {r['separation']:>+8.4f}")

    print(f"\ncross-population references (same instrument, same protocol):")
    for k, v in REFERENCES.items():
        print(f"  {k:<32} {v:.4f}")
    print(f"  chance (70 animals)              {1/70:.4f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"note": "read-only instrument; scored after training, never used for selection",
         "references": REFERENCES, "history": rows}, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
