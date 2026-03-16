import torch
import torchvision.transforms.functional as TF
from equivania.metric import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure


def _rotate_image(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    # Rotate with bilinear interpolation around image center
    return TF.rotate(
        x,
        angle_deg,
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        fill=0,
    )


def register(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    orientation_a_deg: float,
    orientation_b_deg: float,
    *,
    criterion: str = "psnr",
    return_rotation: bool = False,
) -> torch.Tensor:
    """Align ``image_b`` to ``image_a`` using orientation estimates.

    Tries the raw orientation difference and a 180°-shifted alternative, selects the
    rotation giving the higher score w.r.t. ``image_a``. Score uses PSNR by default or
    SSIM when ``criterion='ssim'``.
    """

    # Candidate rotations to bring B into A's frame
    delta = orientation_a_deg - orientation_b_deg

    best_score = None
    best_aligned = None
    best_rotation = None

    def _score(ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        if criterion == "psnr":
            return peak_signal_noise_ratio(ref, est)
        if criterion == "ssim":
            return structural_similarity_index_measure(
                ref.unsqueeze(0), est.unsqueeze(0), data_range=1.0
            )
        raise ValueError(f"Unknown criterion: {criterion}")

    candidate_rotations = [delta, delta + 180.0]
    for rotation in candidate_rotations:
        aligned_b = _rotate_image(image_b, rotation)
        score = _score(image_a, aligned_b)
        if best_score is None or score > best_score:
            best_score = score
            best_aligned = aligned_b
            best_rotation = rotation

    if return_rotation:
        return best_aligned, best_rotation
    return best_aligned
