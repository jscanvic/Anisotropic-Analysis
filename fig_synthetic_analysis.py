import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.von_mises import VonMises

from equivania.angular_profile import compute_angular_profile
from equivania.elementary_signals import elem_sig_isotropic, elem_sig_single_freq_angle
from equivania.gabor_synthesis import gabor_distrib


def wr_cake_wavelet(x):
    theta_vec, power = compute_angular_profile(
        x, sense="counterclockwise", method="cake_wavelet"
    )
    return theta_vec, power / sum(power)


def wr_ridge(x):
    theta_vec, power = compute_angular_profile(
        x, sense="counterclockwise", method="ridge"
    )
    return theta_vec, power / sum(power)


def wr_binning(x):
    theta_vec, power = compute_angular_profile(
        x, sense="counterclockwise", method="binning", n_bins=1000
    )
    inversed = (math.pi - theta_vec) % math.pi
    return inversed, power / sum(power)


def eval_profiles(x, title="", psd_mode_angle=None, von_mises=None):
    stack = []
    stack_theta = []
    stack_names = []

    theta_vec, power = wr_cake_wavelet(x - torch.mean(x))
    stack.append(power)
    stack_theta.append(theta_vec)
    stack_names.append("Cake wavelet")

    theta_vec, power = wr_ridge(x - torch.mean(x))
    stack.append(power)
    stack_theta.append(theta_vec)
    stack_names.append("Ridge")

    theta_vec, power = wr_binning(x - torch.mean(x))
    stack.append(power)
    stack_theta.append(theta_vec)
    stack_names.append("Binning")

    if von_mises is not None:
        theta_vec = torch.tensor([math.pi * s / 1000 for s in range(0, 1000)])
        theta_vec_spectrum = theta_vec + math.pi / 2  # spectrum pi/2 shift
        theta_vmd = 2 * theta_vec_spectrum - math.pi
        density_vmd = torch.exp(von_mises.log_prob(theta_vmd))
        stack.append(density_vmd / sum(density_vmd))
        stack_theta.append(theta_vec)
        stack_names.append("von-Mises")

    return stack, stack_theta, stack_names


def main():
    sz = 256
    print("synthesize image")

    x_iso = elem_sig_isotropic(sz, sz)

    angle_line = np.deg2rad(25)
    angle_spec_line = angle_line
    x_line = elem_sig_single_freq_angle(sz, sz, angle_rad=angle_line)

    y_mode = torch.tensor(np.deg2rad(60))
    angle_spec_gabor = y_mode + math.pi / 2
    vmd_mode = 2 * y_mode - torch.pi  # express in von-Mises domain
    vmd = VonMises(loc=vmd_mode, concentration=torch.tensor(32.0))
    x_gabor, theta_distrib = gabor_distrib(sz, sz, vmd, N=300, seed=1)

    theta_expect = [None, angle_spec_line, angle_spec_gabor]
    x_vec = [x_iso, x_line, x_gabor]

    profile_output_vec = []
    y_max = []
    for x in x_vec:
        out = eval_profiles(x)
        profile_output_vec.append(out)
        power, theta, names = out
        y_max.append(np.max(power))

    y_max = np.max(y_max)
    print(y_max)

    # Get the default property cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    # Extract the colors
    colors = prop_cycle.by_key()["color"]

    color_model = {
        "Binning": colors[2],
        "Ridge": colors[1],
        "Cake wavelet": colors[0],
    }

    opacity_model = {
        "Binning": 0.35,
        "Ridge": 0.9,
        "Cake wavelet": 0.7,
    }

    profiles_fig, ax_vec = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
        subplot_kw={"projection": "polar"},
        sharex=False,
        sharey=False,
        # constrained_layout=True,
    )

    for x, ax, expected, out in zip(
        x_vec, ax_vec, theta_expect, profile_output_vec, strict=False
    ):
        powers, thetas, names = out

        handles = []
        labels = []
        for pow_it, theta_it, name_it in zip(powers, thetas, names, strict=False):
            # symmetrize
            theta_it = torch.concatenate((theta_it, (theta_it + torch.pi)))
            pow_it = torch.concatenate((pow_it, pow_it))

            # then plot
            color = color_model[name_it]
            opacity = opacity_model[name_it]
            (h,) = ax.plot(theta_it, pow_it, label=name_it, color=color, alpha=opacity)
            handles.append(h)
            labels.append(name_it)

        if expected is not None:
            h = ax.axvline(expected, color="red", linestyle="--", linewidth=1)
            handles.append(h)
            labels.append("Expected angle")

        ax.set_rscale("log")
        ax.set_rticks([])  # Fewer radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

    profiles_fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),  # horizontal, vert
        ncol=4,
        frameon=False,
        bbox_transform=profiles_fig.transFigure,
    )

    plt.subplots_adjust(top=0.8)
    plt.savefig("profiles_all.svg")
    plt.savefig("profiles_all.pdf")
    plt.show()


if __name__ == "__main__":
    main()
