import json
import math
import random

import plotly.express as px
import plotly.graph_objects as go
import torch
from paper_fig_synthetic_analysis import wr_cake_wavelet, wr_ridge, wr_binning
from plotly.subplots import make_subplots
from torch.distributions.von_mises import VonMises
from torchvision.transforms.functional import InterpolationMode, rotate

from equivania.gabor_synthesis import gabor_distrib
from equivania.metric import angular_distance, circular_variance, order2_regularity


def equivariance_metric(im, profile_estimator, seed, theta0=0.0):
    angles, power = profile_estimator(im - torch.mean(im))
    mse_tot = 0
    # for k, angle in enumerate(angles):
    M = 5
    for k in range(0, M):
        random.seed(seed + k)
        id = random.randint(0, len(angles) - 1)
        angle = angles[id]

        angle_deg = torch.rad2deg(angle).item()
        interp = InterpolationMode.BILINEAR
        steered_im = rotate(im, angle=angle_deg, interpolation=interp)
        _, steered_power = profile_estimator(steered_im)
        unsteered_steered_power = steered_power.roll(-id)
        mse = (unsteered_steered_power - power).square().mean()
        mse_tot += mse
    return (mse_tot / M).item(), M


def compare_densities(d1, d2):
    # pdf has a sum = 1
    n1 = d1 / sum(d1)
    n2 = d2 / sum(d2)

    # l2 distance (suitable for our case ? smooth decay, error should not be too big even if there is a shift)
    return math.sqrt(sum((n1 - n2) ** 2))


def main():
    sz = 256

    vec_profile_pdf_dist_cake = []
    vec_profile_variance_cake = []
    vec_profile_regularity_cake = []
    vec_profile_equiv_cake = []
    vec_angular_cake = []

    vec_profile_pdf_dist_ridge = []
    vec_profile_variance_ridge = []
    vec_profile_regularity_ridge = []
    vec_profile_equiv_ridge = []
    vec_angular_ridge = []

    vec_profile_pdf_dist_binning = []
    vec_profile_variance_binning = []
    vec_profile_regularity_binning = []
    vec_profile_equiv_binning = []
    vec_angular_binning = []

    repeat = 300
    seed = 0
    for r in range(0, repeat):

        random.seed(seed)
        seed += 1

        rand_v = random.random()
        assert rand_v < 1.0

        y_mode = rand_v * torch.pi  # mode in [0, pi)
        spec_mode = (y_mode + torch.pi / 2) % torch.pi
        # print("spec_mode =", spec_mode)

        vmd_mode = 2 * y_mode - torch.pi  # express in von-Mises domain
        vmd = VonMises(loc=vmd_mode, concentration=torch.tensor(32.0))

        N = 300
        x, theta_distrib = gabor_distrib(sz, sz, vmd, N=N, seed=seed)
        seed += N + 1

        # def gabor_gen(rot):
        #     return gabor_distrib_rot(sz, sz, vmd, N=N, seed=seed + 1, rot=rot)

        theta_vec = torch.tensor([math.pi * s / 1000 for s in range(0, 1000)])
        theta_vec_spectrum = theta_vec + math.pi / 2  # spectrum pi/2 shift
        theta_vmd = 2 * theta_vec_spectrum - math.pi
        density_vec = torch.exp(vmd.log_prob(theta_vmd))

        # -- Cake wavelet
        rad_angles, energies = wr_cake_wavelet(x - torch.mean(x))
        rad_best_angle = rad_angles[torch.argmax(energies)]
        vec_profile_pdf_dist_cake.append(compare_densities(energies, density_vec))
        vec_profile_variance_cake.append(circular_variance(rad_angles, energies))
        vec_profile_regularity_cake.append(order2_regularity(energies))
        eq, M = equivariance_metric(x - torch.mean(x), wr_cake_wavelet, seed)
        vec_profile_equiv_cake.append(eq)
        vec_angular_cake.append(angular_distance(rad_best_angle, spec_mode))
        # print("rad_best_angle =", rad_best_angle)

        # -- Ridge
        rad_angles, energies = wr_ridge(x - torch.mean(x))
        rad_best_angle = rad_angles[torch.argmax(energies)]
        vec_profile_pdf_dist_ridge.append(compare_densities(energies, density_vec))
        vec_profile_variance_ridge.append(circular_variance(rad_angles, energies))
        vec_profile_regularity_ridge.append(order2_regularity(energies))
        eq, M = equivariance_metric(x - torch.mean(x), wr_ridge, seed)
        vec_profile_equiv_ridge.append(eq)
        vec_angular_ridge.append(angular_distance(rad_best_angle, spec_mode))
        # print("rad_best_angle =", rad_best_angle)

        # -- Binning
        rad_angles, energies = wr_binning(x - torch.mean(x))
        rad_best_angle = rad_angles[torch.argmax(energies)]
        vec_profile_pdf_dist_binning.append(compare_densities(energies, density_vec))
        vec_profile_variance_binning.append(circular_variance(rad_angles, energies))
        vec_profile_regularity_binning.append(order2_regularity(energies))
        eq, M = equivariance_metric(x - torch.mean(x), wr_binning, seed)
        vec_profile_equiv_binning.append(eq)
        vec_angular_binning.append(angular_distance(rad_best_angle, spec_mode))
        # print("rad_best_angle =", rad_best_angle)

        seed += M

        # show_img_and_density(img, density_thetas, density_vec)
        print(f"{r+1}/{repeat} ", end="")
    print("")

    metrics_0 = [
        # cake
        vec_profile_pdf_dist_cake,
        vec_profile_variance_cake,
        vec_profile_regularity_cake,
        vec_profile_equiv_cake,
        vec_angular_cake,
        # ridge
        vec_profile_pdf_dist_ridge,
        vec_profile_variance_ridge,
        vec_profile_regularity_ridge,
        vec_profile_equiv_ridge,
        vec_angular_ridge,
        # binning
        vec_profile_pdf_dist_binning,
        vec_profile_variance_binning,
        vec_profile_regularity_binning,
        vec_profile_equiv_binning,
        vec_angular_binning,
    ]

    metrics = []
    for l0 in metrics_0:
        l = []
        for it in l0:
            if isinstance(it, torch.Tensor):
                l.append(it.item())
            else:
                l.append(it)
        metrics.append(l)

    # save
    with open("valid_statistical.json", "w") as json_file:
        json.dump(metrics, json_file)


def imshow(x):
    # in order get the same behavior as pyplot.imshow
    return px.imshow(x.transpose(1, 2).permute(1, 2, 0))


def show_img_and_density(x, thetas, d):
    fig = make_subplots(cols=2)
    fig.add_trace(imshow(x).data[0], row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=thetas,
            y=d,
        ),
        row=1,
        col=2,
    )
    fig.show()


if __name__ == "__main__":
    main()
