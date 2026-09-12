"""At what spatial frequency does the pretrained encoder actually operate?

The acceptance test for the transfer result. Pretraining lifted cross-population
Top-1 from 0.8619 (stock DINOv2) to 0.9143, but that number alone does not say
WHAT the encoder learned, and the answer changes what stage 2 and the rerank
cascade should look like:

  degrades sharply as detail is removed  -> it reads the muzzle PRINT (bead and
                                            ridge topology). The gain is a
                                            resolution play; crops, capture
                                            guidance and LightGlue all compound
                                            with it.
  stays flat                             -> it found some other generalisable
                                            feature (shape, shading, coarse
                                            pattern). Still useful, but the
                                            cascade cannot assume fine detail is
                                            being used, and more resolution buys
                                            nothing.

Method, identical to the original probe: downsample each benchmark image to RxR
and upsample it straight back to its original size before the normal pipeline.
Tensor shapes reaching the encoder never change -- the ONLY thing removed is
detail above frequency R. Where Top-1 breaks is where the model's real operating
frequency is.

Scale anchor: a 518px full frame is ~0.7 mm/px on the muzzle (repo's own
figure), so mm/px at resolution R is 0.7 * 518 / R. Muzzle beads run ~2-5mm and
ridges ~1-3mm, so reading the print at all requires roughly <1.5 mm/px.

The grid keeps every resolution the stock encoders were measured at, so the
curves are directly comparable, and adds points between 256 and 518 because
that is where 0.5-1.0 mm/px actually falls and the original grid jumped
straight from 448 to 224.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from augment import build_val_transform   # noqa: E402
from config import IMG_SIZE               # noqa: E402
from model import GodhaarModel            # noqa: E402
from uk_benchmark.benchmark import evaluate as bench_evaluate  # noqa: E402

# every resolution the stock encoders were probed at, plus 384/320/256/160
GRID = [None, 448, 384, 320, 256, 224, 160, 112, 56, 28, 14]
MM_PER_PX_AT_518 = 0.7

# measured previously on this exact split, same harness (results/probe_freq_*)
STOCK = {
    "production DINOv2 (model.pt)": {
        None: 0.8381, 448: 0.8476, 224: 0.7952, 112: 0.7524,
        56: 0.7381, 28: 0.4429, 14: 0.2143},
    "ResNet50 ImageNet": {
        None: 0.8619, 448: 0.9000, 224: 0.9095, 112: 0.9048,
        56: 0.8524, 28: 0.7667, 14: 0.5810},
}


def degrade(pil: Image.Image, R: int | None) -> Image.Image:
    """Strip detail above R, restore original size. Removes frequency only."""
    if R is None:
        return pil
    w, h = pil.size
    return pil.resize((R, R), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_top1.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "frequency_probe.json"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = GodhaarModel.load_checkpoint(Path(a.checkpoint), device=device)
    model.eval()
    transform = build_val_transform(IMG_SIZE)
    print(f"checkpoint : {Path(a.checkpoint).name}  epoch {ck.get('epoch')}")
    print(f"device     : {device}\n")

    rows = []
    for R in GRID:
        @torch.no_grad()
        def embed_fn(blobs: list[bytes], _R=R):
            imgs = [degrade(Image.open(io.BytesIO(b)).convert("RGB"), _R) for b in blobs]
            x = torch.stack([transform(i) for i in imgs]).to(device)
            e = model(x).float().cpu().numpy()
            return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

        res = bench_evaluate(embed_fn, experiment_name=f"freqprobe_{R or 'full'}",
                             write_result=False)
        mmpx = MM_PER_PX_AT_518 * 518 / (R or 518)
        rows.append({"R": R, "mm_per_px": round(mmpx, 2),
                     "top1": res["top1"], "top5": res["top5"],
                     "separation": res["separation"]})
        print(f"  R={str(R or 'full'):>5}  {mmpx:>6.2f} mm/px   "
              f"top1={res['top1']:.4f}  sep={res['separation']:+.4f}", flush=True)

    base = rows[0]["top1"]
    print(f"\n{'R':>6} {'mm/px':>7} {'top1':>8} {'retained':>9}   "
          + "  ".join(f"{k.split()[0]:>10}" for k in STOCK))
    print("-" * 74)
    for r in rows:
        cells = []
        for name, curve in STOCK.items():
            v = curve.get(r["R"])
            cells.append(f"{v:>10.4f}" if v is not None else f"{'-':>10}")
        print(f"{str(r['R'] or 'full'):>6} {r['mm_per_px']:>7.2f} {r['top1']:>8.4f} "
              f"{r['top1']/base:>8.1%}   " + "  ".join(cells))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"checkpoint": Path(a.checkpoint).name, "epoch": ck.get("epoch"),
         "mm_per_px_at_518": MM_PER_PX_AT_518,
         "method": "downsample to RxR then upsample to original size; only "
                   "frequency above R is removed, tensor shapes unchanged",
         "stock_reference": {k: {str(kk): vv for kk, vv in v.items()}
                             for k, v in STOCK.items()},
         "curve": rows}, indent=2), encoding="utf-8")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
