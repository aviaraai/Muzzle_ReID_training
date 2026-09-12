"""The frequency probe, re-run where the muzzle print is actually resolvable.

The full-frame probe found the pretrained encoder essentially flat from 0.70 to
1.62 mm/px (0.9143 -> 0.9000). Read naively that says "it did not learn the
print". But the benchmark images are FULL FRAMES at ~0.7 mm/px, and this repo's
own figure is that a 1mm ridge gets ~1.4px there -- below Nyquist for the finest
structure. The print is not resolvable in those images even at full resolution,
so a flat curve is what you would measure whether or not the encoder can read
print. The test cannot distinguish the two.

This re-runs the identical sweep in the domain the encoder was TRAINED in:
muzzle crops, ~0.232 mm/px at 518, where a 1mm ridge gets ~4.3px and the print
genuinely is present. If the curve is still flat here, the encoder really is not
using fine detail. If it degrades sharply between roughly 112 and 224 px (0.5-1.0
mm/px in crop space), it reads the print and the full-frame flatness was only
ever a property of the test images.

The instrument is not weakened: bench_evaluate still loads and sha256-verifies
the original benchmark bytes. Cropping happens inside embed_fn, downstream of
verification, exactly as the degrade already does. Crop geometry matches
make_arm2_crops.py (0.60 * min(H,W) square on the pose model's muzzle keypoint),
so this is the same construction the encoder was trained on.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
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

POSE_W = r"D:\Group Projects\inference_server\appstorage\Models\pose_model\best.pt"
CROP_FRAC, KCONF, MUZZLE = 0.60, 0.25, 7
GRID = [None, 448, 384, 320, 256, 224, 160, 112, 80, 56, 28, 14]
MM_PER_PX_AT_518 = 0.232          # crop domain (full frame is 0.70)

_crop_cache: dict[str, Image.Image] = {}


def muzzle_crop(blob: bytes, pose) -> Image.Image:
    """Same crop rule as make_arm2_crops.py. Falls back to the full frame when
    no muzzle keypoint clears KCONF, so the query set stays the full 210 and the
    comparison is not quietly taken over a different population."""
    key = hashlib.sha256(blob).hexdigest()
    if key in _crop_cache:
        return _crop_cache[key]
    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    out = None
    r = pose.predict(bgr, imgsz=640, conf=0.05, verbose=False)[0]
    if r.boxes is not None and len(r.boxes) and r.keypoints is not None \
            and r.keypoints.conf is not None:
        b = int(np.argmax(r.boxes.conf.cpu().numpy()))
        if float(r.keypoints.conf.cpu().numpy()[b][MUZZLE]) >= KCONF:
            cx, cy = r.keypoints.xy.cpu().numpy()[b][MUZZLE]
            h, w = bgr.shape[:2]
            s = int(round(CROP_FRAC * min(h, w)))
            x0 = int(np.clip(cx - s / 2, 0, max(w - s, 0)))
            y0 = int(np.clip(cy - s / 2, 0, max(h - s, 0)))
            out = bgr[y0:y0 + min(s, h), x0:x0 + min(s, w)]
    if out is None:
        out = bgr
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    _crop_cache[key] = pil
    return pil


def degrade(pil: Image.Image, R: int | None) -> Image.Image:
    if R is None:
        return pil
    w, h = pil.size
    return pil.resize((R, R), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_top1.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "frequency_probe_crop.json"))
    a = ap.parse_args()

    from ultralytics import YOLO
    pose = YOLO(POSE_W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = GodhaarModel.load_checkpoint(Path(a.checkpoint), device=device)
    model.eval()
    transform = build_val_transform(IMG_SIZE)
    print(f"checkpoint : {Path(a.checkpoint).name}  epoch {ck.get('epoch')}")
    print(f"domain     : muzzle crops, {MM_PER_PX_AT_518} mm/px at 518\n")

    rows = []
    for R in GRID:
        @torch.no_grad()
        def embed_fn(blobs: list[bytes], _R=R):
            imgs = [degrade(muzzle_crop(b, pose), _R) for b in blobs]
            x = torch.stack([transform(i) for i in imgs]).to(device)
            e = model(x).float().cpu().numpy()
            return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

        res = bench_evaluate(embed_fn, experiment_name=f"freqcrop_{R or 'full'}",
                             write_result=False)
        mmpx = MM_PER_PX_AT_518 * 518 / (R or 518)
        rows.append({"R": R, "mm_per_px": round(mmpx, 3), "top1": res["top1"],
                     "top5": res["top5"], "separation": res["separation"]})
        print(f"  R={str(R or 'full'):>5}  {mmpx:>6.3f} mm/px   "
              f"top1={res['top1']:.4f}  sep={res['separation']:+.4f}", flush=True)

    base = rows[0]["top1"]
    print(f"\n{'R':>6} {'mm/px':>8} {'top1':>8} {'retained':>9}")
    print("-" * 36)
    for r in rows:
        print(f"{str(r['R'] or 'full'):>6} {r['mm_per_px']:>8.3f} {r['top1']:>8.4f} "
              f"{r['top1']/base:>8.1%}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"checkpoint": Path(a.checkpoint).name, "epoch": ck.get("epoch"),
         "domain": "muzzle crop", "mm_per_px_at_518": MM_PER_PX_AT_518,
         "crop": {"frac": CROP_FRAC, "keypoint_conf": KCONF},
         "curve": rows}, indent=2), encoding="utf-8")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
