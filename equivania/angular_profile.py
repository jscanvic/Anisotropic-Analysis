"""Unified angular-profile computations for anisotropy orientation analysis.

Exposes three implementations that return angle centers and associated power:
- Angular PSD (torch-based) matching the logic used in ``main.py``.
- Cake wavelet-like baseline (FFT wedge sweep).
- Ridge baseline.

Each function returns a pair of 1D torch.Tensors: (angles_rad, power).
Angles are expressed in radians.
"""

from __future__ import annotations

import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from equivania.binning import binning
from equivania.bump_mask import MaskKind, apply_bump_mask

Method = Literal["binning", "cake_wavelet", "ridge"]


def compute_cake_wavelet_filter(
    radius: torch.Tensor,
    angle: torch.Tensor,
    *,
    orientation: torch.Tensor,
    resolution: tuple[int, int],
    sigma_theta: float = math.radians(10.0),
    sense: str = "clockwise",
    filtering: str = "bandpass",
) -> torch.Tensor:
    # Cake wavelet wedge centered at theta with radial gating
    if sense == "counterclockwise":
        orientation = -orientation
    elif sense != "clockwise":
        raise ValueError(
            f"Unknown sense '{sense}'. Expected 'clockwise' or 'counterclockwise'."
        )
    H, W = resolution
    diff = torch.arccos(torch.abs(torch.cos(angle - orientation)))
    out = torch.exp(-(diff**2) / (2 * sigma_theta**2))
    if filtering == "bandpass":
        radius_ratio = math.sqrt(H * W)
        low_cutoff = 5 / radius_ratio
        high_cutoff = math.sqrt((H // 2) ** 2 + (W // 2) ** 2) / radius_ratio
        radial_window = (radius > low_cutoff) * (radius < high_cutoff)
        out = radial_window * out
    elif filtering != "allpass":
        raise ValueError(
            f"Unknown filtering '{filtering}'. Expected 'allpass' or 'bandpass'."
        )
    return out


def compute_ridge_filter(
    radius: torch.Tensor,
    angle: torch.Tensor,
    *,
    orientation: torch.Tensor,
    resolution: tuple[int, int],
    sense: str = "counterclockwise",
    filtering: str = "allpass",
) -> torch.Tensor:
    # Directional filter: smoothed line mask defined by -v*sin(theta)+u*cos(theta)=0
    if sense == "clockwise":
        orientation = -orientation
    elif sense != "counterclockwise":
        raise ValueError(
            f"Unknown sense '{sense}'. Expected 'clockwise' or 'counterclockwise'."
        )
    H, W = resolution
    u = radius * torch.sin(angle)
    v = radius * torch.cos(angle)
    out = torch.exp(
        -((u * H * math.cos(orientation) + v * W * math.sin(orientation)).square())
        / 8.0
    )
    if filtering == "bandpass":
        radius_ratio = math.sqrt(H * W)
        low_cutoff = 5 / radius_ratio
        high_cutoff = math.sqrt((H // 2) ** 2 + (W // 2) ** 2) / radius_ratio
        radial_window = (radius > low_cutoff) * (radius < high_cutoff)
        out = radial_window * out
    elif filtering != "allpass":
        raise ValueError(
            f"Unknown filtering '{filtering}'. Expected 'allpass' or 'bandpass'."
        )
    return out


def compute_steerable_profile(
    image: torch.Tensor,
    *,
    angular_resolution: int = 1000,
    filter_kind: str,
    sense: str = "counterclockwise",
    filtering: str = "bandpass",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimates dominant orientation by finding the angle of maximum energy."""
    # Sample angles theta in [0, pi)
    thetas = torch.arange(angular_resolution) * math.pi / angular_resolution

    # Create the grid of frequencies
    H, W = image.shape[-2:]
    u = torch.fft.fftfreq(H, d=1.0, dtype=image.dtype, device=image.device)
    v = torch.fft.fftfreq(W, d=1.0, dtype=image.dtype, device=image.device)
    u, v = torch.meshgrid(u, v, indexing="ij")

    image_ft = torch.fft.fft2(image, dim=(-2, -1))

    # Polar decomposition
    radius = torch.sqrt(u**2 + v**2)
    angle = torch.atan2(u, v)  # Result in [-pi, pi]

    energies = []
    for theta in thetas:
        if filter_kind == "cake_wavelet":
            wedge = compute_cake_wavelet_filter(
                radius,
                angle,
                orientation=theta,
                resolution=(H, W),
                sense=sense,
                filtering=filtering,
            )
        elif filter_kind == "ridge":
            wedge = compute_ridge_filter(
                radius,
                angle,
                orientation=theta,
                resolution=(H, W),
                sense=sense,
                filtering=filtering,
            )
        else:
            raise ValueError(
                f"Unknown filter_kind '{filter_kind}'. Expected 'cake_wavelet' or 'ridge'."
            )
        energy = torch.abs(image_ft * wedge).square().sum()
        energies.append(energy.item())

    energies = torch.tensor(energies, dtype=image.dtype, device=image.device)
    return thetas, energies


def compute_angular_profile(
    x,
    *,
    angular_resolution: int | None = None,
    method: Method = "binning",
    renormalize: bool = False,
    apply_mask: bool = False,
    mask_kind: MaskKind = "unit",
    sense: str | None = None,
    filtering: str | None = None,
    consistent_filtering_binning: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch angular profile computation.

    Parameters
    ----------
    x: torch.Tensor | np.ndarray
        Input image. For torch tensors, expected shape is (C, H, W) or (H, W).
    method: Literal["binning", "cake_wavelet", "ridge"]
        Profile implementation to use.
    renormalize: bool, optional
        If True, L1-normalize the returned power so it sums to 1.
    apply_mask: bool, optional
        If True, apply the unit-disk bump mask e*exp(1/(u^2+v^2-1)) before computing
        the profile (mask is zero outside the unit disk on normalized coordinates).
    kwargs:
        Forwarded to the specific implementation.

    Returns
    -------
    angles_rad: torch.Tensor
        1D tensor of angles in radians.
    power: torch.Tensor
        1D tensor of power/energy per angle.
    """
    if apply_mask:
        x = apply_bump_mask(x, kind=mask_kind)

    if method == "binning":
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if angular_resolution is None:
            if "n_bins" not in kwargs:
                kwargs["n_bins"] = 180
        else:
            if "n_bins" in kwargs:
                raise ValueError("Cannot specify both angular_resolution and n_bins.")
            kwargs["n_bins"] = angular_resolution
        psd_method = kwargs.get("psd_method", "periodogram")
        n_bins = kwargs.get("n_bins", 180)
        positive = kwargs.get("positive", True)
        relative = kwargs.get("relative", False)
        discretization_correction = kwargs.get("discretization_correction", False)
        if consistent_filtering_binning:
            if filtering is None:
                binning_filtering = "bandpass"
            else:
                binning_filtering = filtering
        else:
            binning_filtering = "allpass"
        align_corners = kwargs.get("align_corners", False)
        scale = kwargs.get("scale", "power")
        impl = kwargs.get("impl", "bucketize")
        angles, power = binning(
            x,
            psd_method=psd_method,
            angular_resolution=n_bins,
            positive=positive,
            relative=relative,
            discretization_correction=discretization_correction,
            filtering=binning_filtering,
            align_corners=align_corners,
            scale=scale,
            impl=impl,
        )
    elif method == "cake_wavelet":
        if sense is None:
            sense = "clockwise"
        if filtering is None:
            filtering = "bandpass"
        if angular_resolution is None:
            angular_resolution = 1000
        angles, power = compute_steerable_profile(
            x,
            angular_resolution=angular_resolution,
            filter_kind="cake_wavelet",
            sense=sense,
            filtering=filtering,
        )
    elif method == "ridge":
        if sense is None:
            sense = "counterclockwise"
        if filtering is None:
            filtering = "allpass"
        if angular_resolution is None:
            angular_resolution = 1000
        angles, power = compute_steerable_profile(
            x,
            angular_resolution=angular_resolution,
            filter_kind="ridge",
            sense=sense,
            filtering=filtering,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of binning, cake_wavelet, ridge."
        )

    if renormalize:
        denom = power.sum()
        if torch.is_tensor(denom):
            denom = denom + torch.finfo(power.dtype).eps
        power = power / denom

    return angles, power


if __name__ == "__main__":
    # Generate an oriented anisotropic Gaussian (torch-based)
    h, w = 256, 256
    sigma_x, sigma_y = 8.0, 32.0
    angle_deg = 30.0
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = w / 2.0, h / 2.0
    xr, yr = x - cx, y - cy
    theta = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
    c, s = torch.cos(theta), torch.sin(theta)
    x_rot = c * xr + s * yr
    y_rot = -s * xr + c * yr
    img_t = torch.exp(
        -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
    ).unsqueeze(0)

    # Compute and plot angular profiles for all implementations
    methods = ("binning", "cake_wavelet", "ridge")
    fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(11, 4.5))

    img_np = img_t.squeeze(0).cpu().numpy()
    ax_img.imshow(img_np, cmap="gray")
    ax_img.set_title(f"Anisotropic Gaussian (angle={angle_deg}°)")
    ax_img.axis("off")

    for method in methods:
        angles_rad, power = compute_angular_profile(img_t, method=method)
        angles = torch.as_tensor(angles_rad).detach().cpu().numpy()
        power_np = torch.as_tensor(power).detach().cpu().numpy()
        power_norm = power_np / (power_np.max() + 1e-8)
        ax_prof.plot(np.rad2deg(angles), power_norm, label=method)

    ax_prof.axvline(
        angle_deg, color="k", linestyle="--", linewidth=1.0, label="true angle"
    )
    ax_prof.set_xlabel("Angle (deg)")
    ax_prof.set_ylabel("Normalized power")
    ax_prof.set_title("Angular profiles (example)")
    ax_prof.legend()
    fig.tight_layout()
    plt.show()
