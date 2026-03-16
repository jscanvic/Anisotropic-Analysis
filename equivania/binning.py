import math

import torch


def patchify(x: torch.Tensor, *, patch_size: int, overlap: int):
    """
    Extract the possibly overlapping patches of an image

    :param torch.Tensor x: input image tensor of shape (B, C, H, W)
    :param int patch_size: size of the patches to be extracted
    :param int overlap: number of overlapping pixels between adjacent patches
    :return: tensor of shape (B, N_patches, C, patch_size, patch_size)
    """
    if patch_size <= 0:
        raise ValueError(f"The patch_size must be positive, got {patch_size}")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError(f"The overlap must be in [0, {patch_size}), got {overlap}")
    if x.ndim != 4:
        raise ValueError(f"Input x must have 4 dimensions (B, C, H, W), got {x.ndim}")
    return (
        x.unfold(
            dimension=2, size=patch_size, step=patch_size - overlap
        )  # (B, C, N_h, W, patch_size)
        .unfold(
            dimension=3, size=patch_size, step=patch_size - overlap
        )  # (B, C, N_h, N_w, patch_size, patch_size)
        .permute(0, 2, 3, 1, 4, 5)  # (B, N_h, N_w, C, patch_size, patch_size)
        .flatten(start_dim=1, end_dim=2)  # (B, N_patches, C, patch_size, patch_size)
    )


def compute_psd(x, *, method: str = "periodogram"):
    """
    Compute the PSD of a batch of images

    ..note::

        The PSD is computed per band (channel) independently. It is possible to
        sum over the channels to obtain a pure frequency-power map.

    :param torch.Tensor x: input image of shape (B, C, H, W)
    :param str method: method to compute the PSD, either "periodogram" or "bartlett"
    :return: power spectral density of shape (B, C, H, W)
    """
    if x.ndim != 4:
        raise ValueError(f"Input x must have 4 dimensions (B, C, H, W), got {x.ndim}")
    # Bartlett's method consists in estimting the power spectral density of an
    # image by averaging periodograms of non-overlapping patches, using an
    # unwindowed DFT to compute the periodograms.
    # Welch's method is similar, but uses overlapping patches and a windowed DFT.
    if method in ("bartlett", "welch"):
        patch_size = 32
        if method == "bartlett":
            pass
        elif method == "welch":
            patch_size - 1
        else:
            raise ValueError(f"Unknown method: {method}")
        window = "hann" if method == "welch" else None
    elif method == "periodogram":
        patch_size = None
        window = None
    else:
        raise ValueError(f"Unknown method: {method}")
    if patch_size is not None:
        x_patches = patchify(
            x, patch_size=patch_size, overlap=0
        )  # (B, N_patches, C, patch_size, patch_size)
    else:
        x_patches = x.unsqueeze(1)  # (B, N_patches, C, patch_size, patch_size)
    if window == "hann":
        # Hann window https://en.wikipedia.org/wiki/Window_function#Hann_window
        H, W = x_patches.shape[-2:]
        wh = torch.arange(H, device=x_patches.device, dtype=x_patches.dtype)
        ww = torch.arange(W, device=x_patches.device, dtype=x_patches.dtype)
        wh, ww = torch.meshgrid(wh, ww, indexing="ij")
        w = (
            0.5
            * (1 - torch.cos(2 * math.pi * wh / H))
            * 0.5
            * (1 - torch.cos(2 * math.pi * ww / W))
        )
        x_patches = x_patches * w
    elif window is not None:
        raise ValueError(f"Unknown window: {window}")
    X_patches = torch.fft.fft2(x_patches, norm="ortho")
    psd_patches = X_patches.abs().pow(2)
    psd = psd_patches.mean(dim=1)  # average over patches if any

    # Normalize so that the total power in the psd is the total power of the signal
    psd = x.abs().pow(2).mean() * psd / psd.sum()

    return psd


def binning(
    image: torch.Tensor,
    *,
    angular_resolution: int = 180,
    positive=True,
    relative=False,
    with_angles=True,
    psd_method: str = "periodogram",
    discretization_correction: bool = False,
    impl: str = "bucketize",
    scale: str | None = "power",
    filtering: str = "allpass",
    align_corners: bool = False,
):
    """
    Angular power spectral density from a 2D FFT.
    - n_bins: number of angular bins
    - positive=True: fold antipodal directions -> angles in [0, π)
      (set False for [0, 2π))
    - relative=True: normalize so bins sum to 1 (DC INCLUDED when filtering="allpass")
    - with_angles=True: also return bin-center angles (radians)
    :param bool discretization_correction: Estimate the angular power with an average instead of a sum to account for the different number of samples per angular bin.
    :param str impl: "bucketize" (vectorized) or "mask" (loop with angular masks).
    :param str | None scale: "power" to apply final normalization, None to skip scaling.
    :param str filtering: "allpass" to include DC, "bandpass" to exclude DC.
    :param bool align_corners: If False, zero angle is a bin boundary. If True, zero angle is a bin center.
    """
    # 2D FFT power, sum over all leading dims (e.g., channels/batch)
    psd = compute_psd(image.unsqueeze(0), method=psd_method).squeeze(0)
    lead_dims = tuple(range(psd.ndim - 2))
    psd = psd.sum(dim=lead_dims) if lead_dims else psd  # shape (H, W)

    H, W = psd.shape[-2:]
    device, dtype = psd.device, psd.dtype

    # frequency grid matching fftshift layout
    u = torch.fft.fftfreq(H, d=1.0, device=device)
    v = torch.fft.fftfreq(W, d=1.0, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")

    # angle for each frequency, shift spectrum
    angle = torch.atan2(u, v)  # [-π, π]

    # DC location and mask
    is_dc = (u == 0) & (v == 0)  # after shift, DC at center
    P_dc = psd[is_dc].sum()  # scalar
    mask = ~is_dc  # non-DC

    # angle range / bin edges
    if positive:
        angle = torch.remainder(angle, math.pi)  # [0, π)
        range_max = math.pi
    else:
        angle = torch.remainder(angle + 2 * math.pi, 2 * math.pi)  # [0, 2π)
        range_max = 2 * math.pi

    step = range_max / angular_resolution
    half_step = step / 2
    if align_corners:
        # Shift to make 0 the center of the first bin; wrap for circular bins.
        angle = torch.remainder(angle + half_step, range_max) - half_step
        thetas = torch.arange(angular_resolution, device=device, dtype=dtype) * step
    else:
        thetas = (
            torch.arange(angular_resolution, device=device, dtype=dtype) * step
            + half_step
        )

    apsd = torch.zeros(angular_resolution, device=device, dtype=dtype)
    if impl == "bucketize":
        edges = torch.cat((thetas - half_step, thetas[-1:] + half_step))
        # bin and accumulate non-DC power
        idx = torch.bucketize(angle[mask].reshape(-1).to(dtype), edges) - 1
        idx = idx.clamp_(0, angular_resolution - 1)
        if discretization_correction:
            apsd.scatter_reduce_(0, idx, psd[mask].reshape(-1), reduce="mean")
        else:
            apsd.scatter_add_(0, idx, psd[mask].reshape(-1))
        # spread DC power uniformly across all bins
        if filtering == "allpass":
            apsd += P_dc / angular_resolution
    elif impl == "mask":
        # angular masking sweep, similar to steerable profiles
        for i, theta in enumerate(thetas):
            lower_ok = angle >= theta - half_step if i == 0 else angle > theta - half_step
            bin_mask = mask & lower_ok & (angle <= theta + half_step)
            if discretization_correction:
                count = bin_mask.sum().to(dtype)
                bin_mask = bin_mask.to(dtype) / (count + 1.0)
            else:
                bin_mask = bin_mask.to(dtype)
            if filtering == "allpass":
                bin_mask = bin_mask + is_dc.to(dtype) / angular_resolution
            apsd[i] = (psd * bin_mask).sum()
    else:
        raise ValueError(f"Unknown impl: {impl}. Expected 'bucketize' or 'mask'.")

    # optional normalization
    if scale == "power":
        power = image.abs().pow(2).sum()
        apsd = apsd / apsd.sum() if relative else power * apsd / apsd.sum()
    elif scale is not None:
        raise ValueError(f"Unknown scale: {scale}. Expected 'power' or None.")

    if with_angles:
        return thetas, apsd
    else:
        return apsd
