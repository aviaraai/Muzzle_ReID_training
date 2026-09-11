"""ARM 2 PREPROCESSING - muzzle crops from the 300-identity corpus.

Square of CROP_FRAC * min(H,W) centred on the pose model's muzzle keypoint,
saved at 1024px so augmentation has headroom before the 518 resize.

Identical crop rule to fallback_test.py, so the crops the encoder trains on
are geometrically the same construction the benchmark would be cropped with.

Also collapses the transfer: 17GB of 4000x6000 originals -> ~1GB of crops.
Images with no muzzle keypoint are dropped (8% of this corpus); with a median
of 10 images per identity that costs depth, not identities.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"D:\Group Projects\inference_server")
from ultralytics import YOLO

CORP = Path(r"D:\Group Projects\godhaar-all-images\300-Cattle-source")
OUT = Path(r"D:\Group Projects\godhaar-all-images\300-crops-arm2")
POSE_W = r"D:\Group Projects\inference_server\appstorage\Models\pose_model\best.pt"
CROP_FRAC, KCONF, SAVE_PX, MUZZLE = 0.60, 0.25, 1024, 7

OUT.mkdir(exist_ok=True)
pose = YOLO(POSE_W)
ids = sorted(d.name for d in CORP.iterdir() if d.is_dir())
print(f"{len(ids)} identities", flush=True)

kept, dropped, per_id, confs = 0, 0, {}, {}
for n, i in enumerate(ids):
    d = OUT / i
    d.mkdir(exist_ok=True)
    k = 0
    for p in sorted(CORP.glob(f"{i}/*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            dropped += 1
            continue
        r = pose.predict(im, imgsz=640, conf=0.05, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or r.keypoints.conf is None:
            dropped += 1
            continue
        b = int(np.argmax(r.boxes.conf.cpu().numpy()))
        if r.keypoints.conf.cpu().numpy()[b][MUZZLE] < KCONF:
            dropped += 1
            continue
        _c = float(r.keypoints.conf.cpu().numpy()[b][MUZZLE])
        cx, cy = r.keypoints.xy.cpu().numpy()[b][MUZZLE]
        h, w = im.shape[:2]
        s = int(round(CROP_FRAC * min(h, w)))
        x0 = int(np.clip(cx - s / 2, 0, max(w - s, 0)))
        y0 = int(np.clip(cy - s / 2, 0, max(h - s, 0)))
        crop = im[y0:y0 + min(s, h), x0:x0 + min(s, w)]
        crop = cv2.resize(crop, (SAVE_PX, SAVE_PX), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / p.name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        confs[f"{i}/{p.name}"] = _c
        kept += 1
        k += 1
    per_id[i] = k
    if (n + 1) % 25 == 0:
        print(f"  {n+1}/{len(ids)} identities  kept={kept} dropped={dropped}", flush=True)

cnt = np.array(list(per_id.values()))
size_gb = sum(f.stat().st_size for f in OUT.rglob("*.jpg")) / 1e9
print(f"\n=== ARM 2 CROPS ===")
print(f"  kept {kept} / {kept+dropped} ({100*kept/(kept+dropped):.1f}%)  dropped {dropped}")
print(f"  identities with >=6 crops : {int((cnt>=6).sum())}/{len(cnt)}")
print(f"  identities with <3 crops  : {int((cnt<3).sum())}")
print(f"  per-identity: min={cnt.min()} median={int(np.median(cnt))} max={cnt.max()}")
print(f"  on disk: {size_gb:.2f} GB  (source was 17 GB)")
newly = sum(1 for v in confs.values() if v < 0.50)
print(f"  newly recovered (conf 0.25-0.50): {newly}")
(OUT / "_ARM2_CROP_MARKER.json").write_text(json.dumps({
    "crop_frac": CROP_FRAC, "keypoint_conf_threshold": KCONF, "save_px": SAVE_PX,
    "generated_by": "make_arm2_crops.py",
    "note": "square crop centred on pose-model muzzle keypoint; every image "
            "under this root is cropped",
}, indent=2), encoding="utf-8")
# train_dinov2_arcface.py --arm crop checks for THIS FILE, not a path
# substring -- a substring check breaks the moment the corpus is bind-mounted
# or copied somewhere whose name doesn't happen to contain "crop".
print(f"  wrote _ARM2_CROP_MARKER.json (required by --arm crop)")

json.dump({"crop_frac": CROP_FRAC, "save_px": SAVE_PX, "keypoint_conf": KCONF,
           "kept": kept, "dropped": dropped, "per_identity": per_id,
           "newly_recovered": newly, "confs": confs,
           "size_gb": size_gb, "out": str(OUT)},
          open(Path(__file__).with_name("arm2_crops.json"), "w"), indent=2)
print(f"\nwrote {OUT}")
