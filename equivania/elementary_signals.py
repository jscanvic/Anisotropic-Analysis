import torch


def elem_sig_isotropic(w, h, freq=8, sigma=0.2, device="cpu"):
    decay = 1 / sigma

    # 1. Create a coordinate grid from -1 to 1
    # steps=size ensures we have the correct resolution
    coords_y = torch.linspace(-1, 1, steps=h, device=device)
    coords_x = torch.linspace(-1, 1, steps=w, device=device)
    grid_x, grid_y = torch.meshgrid(coords_x, coords_y, indexing="ij")

    # 2. Calculate radial distance r
    r = torch.sqrt(grid_x**2 + grid_y**2)

    # 3. Calculate the decaying sinusoid
    pattern = torch.sin(2 * torch.pi * freq * r) * torch.exp(-decay * r)
    pattern = 0.5 + pattern / 2

    im = pattern.unsqueeze(0)
    im = im.expand(3, *im.shape[1:])
    return im


def elem_sig_single_freq_angle(
    w, h, angle_rad=torch.pi / 5, freq=8, sigma=0.05, device="cpu"
):
    # 1. Create a coordinate grid from -1 to 1
    # steps=size ensures we have the correct resolution
    coords_y = torch.linspace(-1, 1, steps=h, device=device)
    coords_x = torch.linspace(-1, 1, steps=w, device=device)

    # since zero in images is at the top, we take opposite sign
    coords_y = -coords_y

    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")

    # 2. Rotate coordinates to align with the specified angle
    # u is distance along the line, v is distance perpendicular to the line
    cos_t = torch.cos(torch.tensor(angle_rad))
    sin_t = torch.sin(torch.tensor(angle_rad))

    u = grid_x * cos_t + grid_y * sin_t

    # orthogonal axis
    v = -grid_x * sin_t + grid_y * cos_t  # wedge

    # create sinusoid along specified angle
    sin_img = torch.sin(2 * torch.pi * freq * u)

    # smooth line width
    gaussian_width = torch.exp(-(v**2) / (2 * sigma**2))

    pattern = 0.5 + sin_img * gaussian_width / 2
    im = pattern.unsqueeze(0)
    im = im.expand(3, *im.shape[1:])
    return im
