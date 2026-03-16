import math

import torch


def angular_distance(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """Angular distance between two angles"""
    return torch.acos(torch.cos(angle_a - angle_b))


def circular_variance(
    angles: torch.Tensor,
    weights: torch.Tensor,
    *,
    dim: int = -1,
    oriented: bool = False,
) -> torch.Tensor:
    """Compute the circular variance of a weighted sum of finitely-many Dirac delta functions"""
    weights = weights / weights.sum(dim=dim)
    real_part = torch.sum(weights * angles.cos(), dim=dim)
    if oriented:
        imaginary_part = (weights * angles.sin()).sum(dim=dim)
        r = torch.sqrt(real_part**2 + imaginary_part**2)
    else:
        r = real_part.abs()
    return 1.0 - r


def order2_regularity(samples, dx=None):
    """
    Estimation of L2 norm with finite differences

    :param samples: Sampling of the function
    :param dx: Sampling step, default value = math.pi/len(samples)
    """
    if dx is None:
        dx = math.pi / len(samples)

    f = samples
    d2_estimation = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2

    return math.sqrt(sum([y**2 for y in d2_estimation]))


def peak_signal_noise_ratio(
    reference: torch.Tensor, estimate: torch.Tensor, *, peak: float = 1.0
) -> torch.Tensor:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between reference and estimate"""
    height, width = reference.shape[-2:]
    u = torch.arange(height, device=reference.device, dtype=reference.dtype)
    v = torch.arange(width, device=reference.device, dtype=reference.dtype)
    u, v = torch.meshgrid(u, v, indexing="ij")
    ch, cw = (height - 1) / 2.0, (width - 1) / 2.0
    radius = min(height, width) / 2.0
    mask = (u - ch) ** 2 + (v - cw) ** 2 <= radius**2
    mask = mask.view((1,) * (reference.ndim - 2) + mask.shape)
    mask = mask.repeat(*reference.shape[:-2], 1, 1)

    # We're happy with a solution that's good up to a 180-degree rotation so we test both rotated and unrotated versions
    psnr = None
    for rotated in (False, True):
        candidate_estimate = (
            torch.flip(estimate, dims=(-2, -1)) if rotated else estimate
        )
        err = reference - candidate_estimate
        err = err[mask]
        mse = err.square().mean()
        candidate_psnr = 10 * torch.log10((peak**2) / mse)
        psnr = candidate_psnr if psnr is None else torch.maximum(psnr, candidate_psnr)

    return psnr
