import math
import random

import torch
from skimage.filters import gabor_kernel
from torchvision.transforms import ToTensor


def project_to_circle(i, j, r, cx, cy):
    """
    :param i: image index
    :param j: image index
    :param r: circle radius
    :param cx: center width
    :param cy: center height
    """
    dx, dy = i - cx, j - cy
    d = (dx**2 + dy**2) ** 0.5
    if d <= r:
        return (i, j)  # Point is already inside or on the circle
    else:
        ux, uy = dx / d, dy / d
        x_proj = cx + r * ux
        y_proj = cy + r * uy
        return (x_proj, y_proj)


def rotate_point(i, j, w, h, theta):
    cx, cy = w / 2, h / 2
    # Translate to origin
    x, y = i - cx, j - cy
    # Apply rotation
    x_rot = x * math.cos(theta) - y * math.sin(theta)
    y_rot = x * math.sin(theta) + y * math.cos(theta)
    # Translate back
    i_rot = int(x_rot + cx)
    j_rot = int(y_rot + cy)
    return (i_rot, j_rot)


def gabor_distrib_rot(
    w, h, angle_distrib, N=50, lambd=None, seed=0, rescale=True, rot=0
) -> torch.Tensor:
    """
    Docstring for gabor_distrib

    :param w: width
    :param h: height
    :param angle_distrib: distribution of angles (in radiant)
    :param N: number of atoms
    :param lambd: used if N is None, poisson related atom covering count
    :param seed: digital identifier for reproducibility
    :param rescale: set pixel values in [0, 1]
    :return: Description
    :rtype: Tensor
    """

    if N is None:
        if lambd is None:
            raise ValueError("Either N or lambd must be specified")
        N = torch.poisson(torch.tensor(h * w * lambd).float()).long().item()

    im = torch.zeros(3, w, h) + 1 / 2

    random.seed(seed)
    torch.manual_seed(seed)
    seed += 1
    theta_vec = (angle_distrib.sample((N,)) + torch.pi) / 2  # [-pi, pi) to [0, pi)

    for iter in range(0, N):
        random.seed(seed + iter)
        sigma = random.randint(1, 8) * math.pi / 2
        frequency = 0.025 * random.randint(3, 8)
        amp = 0.05 + (0.1 * random.randint(0, 3))

        ker = gabor_kernel(
            frequency, theta=(theta_vec[iter] - rot), sigma_x=sigma, sigma_y=sigma
        )
        r = ker.shape[-1]

        ir = random.randint(r // 2 + 1, w - r // 2 - 1)
        jr = random.randint(r // 2 + 1, h - r // 2 - 1)
        cx = w / 2
        cy = h / 2

        ir, jr = project_to_circle(ir, jr, w // 2 - r // 2 - 1, cx=cx, cy=cy)
        i0, j0 = rotate_point(ir, jr, w, h, rot)
        # i0 = int((ir - (w-1)/2) * math.cos(rot) - (jr - (h-1)/2) * math.sin(rot) + w/2)
        # j0 = int((ir - (w-1)/2) * math.sin(rot) + (jr - (h-1)/2) * math.cos(rot) + h/2)

        ker = ToTensor()(ker)
        ker = torch.real(ker)
        ker = ker / ker.max() * amp
        im[
            :,
            (i0 - r // 2) : (i0 + r // 2 + (r % 2)),
            (j0 - r // 2) : (j0 + r // 2 + (r % 2)),
        ] += ker

    if rescale:
        im = im - im.min()
        im = im / im.max()

    return im.transpose(1, 2), theta_vec


def gabor_distrib(
    w, h, angle_distrib, N=50, lambd=None, seed=0, rescale=True
) -> torch.Tensor:
    """
    Docstring for gabor_distrib

    :param w: width
    :param h: height
    :param angle_distrib: distribution of angles (in radiant)
    :param N: number of atoms
    :param lambd: used if N is None, poisson related atom covering count
    :param seed: digital identifier for reproducibility
    :param rescale: set pixel values in [0, 1]
    :return: Description
    :rtype: Tensor
    """

    if N is None:
        if lambd is None:
            raise ValueError("Either N or lambd must be specified")
        N = torch.poisson(torch.tensor(h * w * lambd).float()).long().item()

    im = torch.zeros(3, w, h) + 1 / 2

    random.seed(seed)
    torch.manual_seed(seed)
    seed += 1
    theta_vec = (angle_distrib.sample((N,)) + torch.pi) / 2  # [-pi, pi) to [0, pi)

    for iter in range(0, N):
        random.seed(seed + iter)
        sigma = random.randint(1, 8) * math.pi / 2
        frequency = 0.025 * random.randint(3, 8)
        amp = 0.05 + (0.1 * random.randint(0, 3))

        ker = gabor_kernel(
            frequency, theta=theta_vec[iter], sigma_x=sigma, sigma_y=sigma
        )
        r = ker.shape[-1]
        i0 = random.randint(r // 2 + 1, w - r // 2 - 1)
        j0 = random.randint(r // 2 + 1, h - r // 2 - 1)

        ker = ToTensor()(ker)
        ker = torch.real(ker)
        ker = ker / ker.max() * amp
        im[
            :,
            (i0 - r // 2) : (i0 + r // 2 + (r % 2)),
            (j0 - r // 2) : (j0 + r // 2 + (r % 2)),
        ] += ker

    if rescale:
        im = im - im.min()
        im = im / im.max()

    return im.transpose(1, 2), theta_vec


def synthesize_gabor(w, h, N=50):
    im = torch.zeros(3, w, h) + 1 / 2

    for k in range(0, N):
        sigma = random.randint(1, 8) * math.pi / 2
        frequency = 0.025 * random.randint(3, 8)
        theta = math.pi / 16 * random.randint(1, 10)
        amp = 0.05 + (0.1 * random.randint(0, 3))

        k = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
        r = k.shape[-1]
        i0 = random.randint(r // 2 + 1, w - r // 2 - 1)
        j0 = random.randint(r // 2 + 1, h - r // 2 - 1)

        k = ToTensor()(k)
        k = torch.real(k)
        k = k / k.max() * amp
        im[
            :,
            (i0 - r // 2) : (i0 + r // 2 + (r % 2)),
            (j0 - r // 2) : (j0 + r // 2 + (r % 2)),
        ] += k

    im = im - im.min()
    im = im / im.max()

    return im
