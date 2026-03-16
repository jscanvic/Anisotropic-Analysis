from __future__ import annotations

import math
from typing import Literal

import torch

MaskKind = Literal["unit", "radial"]


def make_bump_mask(
    h: int, w: int, *, kind: MaskKind, device=None, dtype=None
) -> torch.Tensor:
    """Build a 2D bump mask on normalized coordinates for the requested flavor."""
    if h <= 0 or w <= 0:
        raise ValueError("Mask dimensions must be positive")
    v = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    u = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(v, u, indexing="ij")
    if kind == "unit":
        r2 = xx * xx + yy * yy
        mask = torch.zeros_like(r2)
        inside = r2 < 1.0
        mask[inside] = math.e * torch.exp(1.0 / (r2[inside] - 1.0))
        return mask
    if kind == "radial":
        rr = torch.sqrt(xx * xx + yy * yy)
        base = torch.exp(-4.0 * (rr.clamp(max=1.0) ** 2))
        return base * (rr <= 1.0)
    raise ValueError(f"Unknown mask kind: {kind}")


def apply_bump_mask(x: torch.Tensor, *, kind: MaskKind) -> torch.Tensor:
    """Apply the requested bump mask variant to a 2D/3D tensor image."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Unsupported type {type(x)} for masking (expected torch.Tensor)"
        )
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    mask = make_bump_mask(h, w, kind=kind, device=x.device, dtype=x.dtype)
    if x.ndim == 2:
        return x * mask
    return x * mask.unsqueeze(0)


if __name__ == "__main__":
    # Visualize both bump mask flavors on a 128x128 grid for quick inspection
    size = 128
    masks = {k: make_bump_mask(size, size, kind=k) for k in ("unit", "radial")}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        for kind, mask in masks.items():
            stats = mask.min().item(), mask.max().item(), mask.mean().item()
            print(f"{kind} mask stats (min, max, mean): {stats}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (kind, mask) in zip(axes, masks.items(), strict=False):
            im = mask.detach().cpu().numpy()
            mappable = ax.imshow(im, cmap="inferno")
            ax.set_title(f"{kind.capitalize()} mask")
            ax.axis("off")
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Bump masks on {size}x{size} grid")
        fig.tight_layout()
        plt.show()
