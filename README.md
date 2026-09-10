# Godhaar muzzle re-id — DINOv2 + Sub-center ArcFace fine-tuning

Everything needed to run is already in this folder, including the images.
There is no reconstruction step and nothing to download or copy in.

## Run it

Requires: NVIDIA driver + Docker + nvidia-container-toolkit on the host.
Nothing else — no Python, no uv, no CUDA toolkit.

```bash
docker compose build      # first time, and after ANY file in this folder changes
docker compose up
```

That's it. `--gpu-preset auto` detects the card and picks the batch size,
precision and freeze schedule for it. On an A6000 that resolves to
batch=32, grad-accum=1, bfloat16.

**`docker compose build` is not optional after replacing files.** The
Dockerfile copies `scripts/`, `main.py` and `data/` *into* the image, so a
plain `docker compose up` keeps running whatever was baked in at the last
build — this is the single most common way to end up debugging a bug that
was already fixed.

## Output

Both directories are mounted as volumes, so they survive `docker compose
down` and container replacement:

- `checkpoints/` — `last.pt`, `best_top1.pt`, `best_gap.pt`
- `results/` — `metrics.csv`, `training.log`, `loss_curves.png`

Watch `Top1` and `Gap` per epoch in the log. `Gap` is the separation
between genuine and impostor similarity — it matters more than raw Top-1
for production thresholding, which is why it gets its own checkpoint.

## Options

```bash
docker compose run --rm train --epochs 80
docker compose run --rm train --gpu-preset a6000 --batch-size 48
docker compose run --rm train --resume          # continue from checkpoints/last.pt
```

## The data

`data/CattleMuzzle/<CattleID>/<Image>` — 388 images across 121 identities,
listed in `data/benchmark_manifest.csv` (with a sha256 per file).

Selection was done by running each candidate image through the *production*
quality and crop gate (`quality_check` + `crop_cattle`), not by eye, plus
SHA-256 duplicate detection. Muzzle, front and left photos are all eligible;
the gate alone decides. Identities with fewer than 3 surviving images are
dropped, since ArcFace needs several examples per class.

Train/val is **identity-disjoint**: the 18 validation identities appear
nowhere in the 103 training identities. This is deliberate — production
does open-set matching against animals the encoder never trained on, so a
closed-set split would report a number that doesn't transfer. It's also why
`evaluate()` does leave-one-out retrieval *within* the query set rather than
querying against the gallery.
