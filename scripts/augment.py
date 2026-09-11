"""Augmentation for the texture-encoder pretraining run.

Every choice here is calibrated against something measured on the benchmark
rather than taken from a default recipe:

  * NO GaussianBlur. The previous pipeline applied sigma up to 3.0 at p=0.4.
    That destroys the exact band this experiment exists to measure -- with
    everything coarser than ~10px removed, genuine and impostor pairs still
    separate at AUC 0.8909 (probe B), so the ridge band carries real identity
    signal and blurring it away trains the model to ignore it.

  * NO RandomResizedCrop. Its scale=(0.7,1.0) ratio=(0.6,1.5) manufactured
    aspect distortion far outside the measured range: benchmark aspect ratio
    sits at median 0.918, IQR [0.781, 1.088]. Replaced by a mild RandomAffine.

  * RandomAffine degrees=8, matched to the measured roll distribution --
    median |roll| 3.9 deg, p90 9.2 deg, max 16.9 deg, 96.7% within +/-15 deg.
    Rotating further than the data ever rotates teaches invariance nobody
    needs.

  * RandomErasing shrunk from scale (0.05,0.25) to (0.02,0.12). 64% of
    benchmark images already carry a real occlusion (feed 26.6%, hand 20.8%,
    halter rope 16.2%), so modelling occlusion is right -- but 25% patches
    wipe out too much of the ridge field to learn from what remains.

  * JPEG and noise added: both model genuine capture variation, and neither
    is scale-selective the way blur is.

  * No horizontal flip. A mirrored muzzle print is a different pattern, not
    the same animal.
"""
from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JPEGCompression(torch.nn.Module):
    """Re-encode through JPEG at a random quality. Operates on a float tensor
    in [0,1] and returns one, so it can sit after ToTensor() alongside the
    other tensor-space transforms."""

    def __init__(self, quality: tuple[int, int] = (40, 95)) -> None:
        super().__init__()
        self.qmin, self.qmax = quality

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        q = int(torch.randint(self.qmin, self.qmax + 1, (1,)).item())
        arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = torch.from_numpy(
            __import__("numpy").array(Image.open(buf).convert("RGB"))
        ).permute(2, 0, 1).float() / 255.0
        return out.to(img.dtype)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(quality=({self.qmin}, {self.qmax}))"


class GaussianNoise(torch.nn.Module):
    """Additive Gaussian sensor noise, clamped back into [0,1]."""

    def __init__(self, sigma: tuple[float, float] = (0.01, 0.04)) -> None:
        super().__init__()
        self.smin, self.smax = sigma

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        s = torch.empty(1).uniform_(self.smin, self.smax).item()
        return (img + torch.randn_like(img) * s).clamp(0, 1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sigma=({self.smin}, {self.smax}))"


def build_train_transform(img_size: int) -> transforms.Compose:
    # Resize to a margin, apply the affine there, then centre-crop back to
    # img_size. Applying the affine directly at img_size leaves black corners
    # in almost every sample -- caught by eye on the first augmentation grid --
    # a systematic artifact the model can key on and one the val transform
    # never produces.
    #
    # 1.35 rather than a tighter margin: the affine can shrink by 5% AND rotate
    # 8 degrees AND translate 3%, and those compound. The largest axis-aligned
    # square inside a square of side S rotated by theta is S/(cos+sin) = S/1.129,
    # so the safe region is 0.95*m/1.129 - 0.06*m = 0.781*m, which needs
    # m >= 663 to cover a 518 crop. A first attempt at 1.15 failed the corner
    # test; the number is verified by test_train_transform_leaves_no_black_corners
    # rather than by this arithmetic.
    margin = int(round(img_size * 1.35))
    return transforms.Compose([
        transforms.Resize((margin, margin)),
        transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.RandomApply([JPEGCompression(quality=(40, 95))], p=0.3),
        transforms.RandomApply([GaussianNoise(sigma=(0.01, 0.04))], p=0.3),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])


def build_val_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
