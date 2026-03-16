import math
from collections.abc import Callable

import torch
import torchvision.transforms.functional as TF


def angular_distance(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """Oriented angular distance between two angles in radians, in [0, pi]."""
    return torch.acos(torch.cos(angle_a - angle_b))


def _rotate_image(image: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    angle_deg = torch.rad2deg(angle_rad).item()
    return TF.rotate(
        image,
        -angle_deg,
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        fill=0,
    )


def equivariance_metric(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    register_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_angles: int = 5,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute equivariance error for a registration module under shared rotations.

    The registration function must return the CCW rotation (in radians) that maps
    ``image_b`` to ``image_a``.
    """
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")

    device = image_a.device
    dtype = image_a.dtype
    base_theta = torch.as_tensor(
        register_fn(image_a, image_b), device=device, dtype=dtype
    )

    if seed is None:
        deltas = torch.rand(num_angles, device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        deltas = torch.rand(num_angles, generator=generator, device=device, dtype=dtype)
    deltas = deltas * (2.0 * math.pi)

    distances = []
    for delta in deltas:
        rotated_a = _rotate_image(image_a, delta)
        rotated_b = _rotate_image(image_b, delta)
        theta_delta = torch.as_tensor(
            register_fn(rotated_a, rotated_b), device=device, dtype=dtype
        )
        distances.append(angular_distance(theta_delta, base_theta))

    return torch.stack(distances).mean()
