import matplotlib.colors as colors

# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import vflip
from torch.distributions.von_mises import VonMises
from torch.fft import fft2, fftshift

from equivania.elementary_signals import elem_sig_isotropic, elem_sig_single_freq_angle
from equivania.gabor_synthesis import gabor_distrib


def save_logplot(data_in, title, v_min, v_max):
    # data_in has y_axis zero at bottom
    # since we save it as an image we flip it to set it on the top
    data = vflip(data_in).numpy()

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height]
    ax.set_aspect("equal")
    ax.axis("off")
    plt.pcolor(
        range(256),
        range(256),
        data,
        cmap="gray",
        shading="nearest",
        norm=colors.LogNorm(vmin=v_min, vmax=v_max),
    )

    plt.savefig(title, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    sz = 256
    print("synthesize image")

    angle_rad = np.deg2rad(25)
    x = elem_sig_single_freq_angle(sz, sz, angle_rad=angle_rad)
    torchvision.io.write_png(
        (x * 255).to("cpu", torch.uint8), "synthetic_oscillating_line.png",
    )
    fx_line = torch.abs(fftshift(fft2(x, norm="ortho"), dim=[-2,-1]))
    # torchvision.io.write_png(
    #     (fx_line * 255).to("cpu", torch.uint8), "synthetic_oscillating_line_s0.png",
    # )
    fx_line = torch.sum(fx_line, dim=0) / 3

    x = elem_sig_isotropic(sz, sz)
    torchvision.io.write_png(
        (x * 255).to("cpu", torch.uint8), "synthetic_isotropic.png"
    )
    fx_iso = torch.abs(fftshift(fft2(x, norm="ortho")))
    fx_iso = torch.sum(fx_iso, dim=0) / 3

    y_mode = torch.tensor(np.deg2rad(60))
    vmd_mode = 2 * y_mode - torch.pi  # express in von-Mises domain
    vmd = VonMises(loc=vmd_mode, concentration=torch.tensor(32.0))
    x, theta_distrib = gabor_distrib(sz, sz, vmd, N=300, seed=1)
    torchvision.io.write_png(
        (x * 255).to("cpu", torch.uint8), "synthetic_gabor.png"
    )
    fx_gabor = torch.abs(fftshift(fft2(x, norm="ortho")))
    fx_gabor = torch.sum(fx_gabor, dim=0) / 3

    vmin = np.min([fx_iso.min(), fx_line.min(), fx_gabor.min()]) + 0.00001
    vmax = np.min([fx_iso.max(), fx_line.max(), fx_gabor.max()])

    save_logplot(fx_iso, "synthetic_isotropic_spec.png", vmin, vmax)
    save_logplot(fx_line, "synthetic_oscillating_line_spec.png", vmin, vmax)
    save_logplot(fx_gabor, "synthetic_gabor_spec.png", vmin, vmax)


if __name__ == "__main__":
    main()
