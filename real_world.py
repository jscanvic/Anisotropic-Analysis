import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import yaml
from equivania.angular_profile import (
    compute_angular_profile,
    compute_cake_wavelet_filter,
    compute_ridge_filter,
)
from equivania.bump_mask import apply_bump_mask
from equivania.equiv import equivariance_metric
from equivania.metric import angular_distance, circular_variance, peak_signal_noise_ratio
from PIL import Image
from equivania.registration import register
from skimage import data as skdata
from equivania.steerable_image import SteerableImage
from equivania.tomography import load_scan
from torchmetrics.functional import structural_similarity_index_measure

_OPEN_FIGURES: list[plt.Figure] = []

PLOT_PARAMS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "text.usetex": True,
}
plt.rcParams.update(PLOT_PARAMS)
plt.ion()

config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument(
    "-c",
    "--config",
    type=Path,
    help="YAML config file providing CLI defaults (later CLI overrides).",
)

# Set up argparse
parser = argparse.ArgumentParser(
    description="Process an image and compare angular profiles."
)
parser.add_argument("--im_kind", type=str, default="tomography")
parser.add_argument(
    "--dicom_file", type=str, default="ct_scan.dcm", help="Path to the DICOM file."
)
parser.add_argument(
    "--texture_path",
    type=str,
    default="texture.jpg",
    help="Path to the texture image when im_kind='texture'.",
)
parser.add_argument(
    "--angular_resolution",
    type=int,
    default=None,
    help="Number of angle bins in the angular profile",
)
parser.add_argument(
    "--psd_method",
    type=str,
    default="welch",
    choices=["periodogram", "bartlett", "welch"],
    help="Method to compute the PSD.",
)
# make it possible to undo with --no-discretization_correction
parser.add_argument(
    "--discretization_correction",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--apply_mask",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply unit bump mask before angular profile computation.",
)
parser.add_argument(
    "--mask_kind",
    type=str,
    default="unit",
    choices=["unit", "radial"],
    help="Shape of bump mask to use when --apply_mask is enabled.",
)
parser.add_argument(
    "--display_mask_height",
    type=int,
    default=256,
    help="Height used to render the angular filter masks.",
)
parser.add_argument(
    "--display_mask_width",
    type=int,
    default=256,
    help="Width used to render the angular filter masks.",
)
parser.add_argument(
    "--display_mask_orientation",
    type=float,
    default=0.0,
    help="Orientation in degrees used when rendering the angular filter masks.",
)
parser.add_argument(
    "--apply_preprocessing_mask",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Apply radial bump mask after the mandatory center crop when building the SteerableImage. "
        "Defaults to enabled for texture/scikit inputs."
    ),
)
parser.add_argument(
    "--profile_methods",
    nargs="+",
    choices=["binning", "cake_wavelet", "ridge"],
    default=["binning", "ridge", "cake_wavelet"],
    help="Angular profile methods to compare.",
)
parser.add_argument(
    "--angular_profile_normalization",
    type=str,
    default="normalized",
    help='Angular profile normalization mode: "normalized" or "unnormalized".',
)
parser.add_argument(
    "--angular_profile_sense",
    type=str,
    default=None,
    help=(
        "Optional sense override forwarded to compute_angular_profile (e.g. clockwise/counterclockwise)."
    ),
)
parser.add_argument(
    "--angular_profile_filtering",
    type=str,
    default=None,
    help=(
        "Optional filtering override forwarded to compute_angular_profile (e.g. bandpass/allpass)."
    ),
)
parser.add_argument(
    "--consistent_filering_binning",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use consistent filtering when computing binning profiles.",
)
# orientation options
parser.add_argument(
    "--base_orientation_a",
    type=float,
    default=30.0,
    help="Base orientation of the first image (degrees).",
)
parser.add_argument(
    "--base_orientation_b",
    type=float,
    default=140.0,
    help="Base orientation of the second image (degrees).",
)
parser.add_argument(
    "--sweep_steps",
    type=int,
    default=37,
    help="Number of angles in the sweep validation (0° inclusive, 180° exclusive).",
)
parser.add_argument(
    "--scikit_image",
    type=str,
    default="astronaut",
    choices=[
        "astronaut",
        "camera",
        "chelsea",
        "coffee",
        "horse",
        "rocket",
        "cat",
        "text",
    ],
    help="Which scikit-image sample to load when im_kind='scikit'.",
)
parser.add_argument(
    "--registration_criterion",
    type=str,
    default="ssim",
    choices=["psnr", "ssim"],
    help="Score used to select best alignment (psnr or ssim).",
)
parser.add_argument(
    "--registration_metric",
    type=str,
    default="psnr",
    choices=["psnr", "ssim"],
    help=(
        "Metric used for reporting/analysis after registration; does not affect alignment."
    ),
)
parser.add_argument(
    "--equivariance_angles",
    type=int,
    default=5,
    help="Number of random angles used to estimate equivariance error.",
)
parser.add_argument(
    "--equivariance_seed",
    type=int,
    default=0,
    help="Random seed for equivariance error sampling (default: none).",
)
parser.add_argument(
    "--hold_plots",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep plot windows open until closed (default: enabled).",
)

config_ns, remaining = config_parser.parse_known_args()
if config_ns.config:
    config_path = config_ns.config.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a top-level mapping")
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        if action.dest in config_data and config_data[action.dest] is not None:
            parser.set_defaults(**{action.dest: config_data[action.dest]})
args = parser.parse_args(remaining)


def _profile_kwargs(method: str) -> dict:
    if method == "binning":
        return dict(
            psd_method=args.psd_method,
            discretization_correction=args.discretization_correction,
            positive=True,
            relative=False,
            impl="mask",
            align_corners=True,
            scale=None,
            consistent_filtering_binning=args.consistent_filering_binning,
        )
    return {}


def _renormalize_flag() -> bool:
    if args.angular_profile_normalization == "normalized":
        return True
    if args.angular_profile_normalization == "unnormalized":
        return False
    raise ValueError(
        f"Unknown angular_profile_normalization: {args.angular_profile_normalization}"
    )


def _center_crop(img: torch.Tensor) -> torch.Tensor:
    # Enforce even square crop so spatial statistics are comparable across inputs
    size = min(2 * (img.shape[-2] // 2), 2 * (img.shape[-1] // 2))
    return TF.center_crop(img, size)


def _should_apply_preprocessing_mask(kind: str) -> bool:
    if args.apply_preprocessing_mask is not None:
        return args.apply_preprocessing_mask
    return kind in {"texture", "scikit"}


def _preprocess_tensor(kind: str, img: torch.Tensor) -> torch.Tensor:
    if img is None:
        return img
    cropped = _center_crop(img)
    if _should_apply_preprocessing_mask(kind):
        return apply_bump_mask(cropped, kind=args.mask_kind)
    return cropped


def _make_image(kind: str, filepath: str, base_orientation: float) -> SteerableImage:
    match kind:
        case "tomography":
            image = load_scan(filepath)
        case "texture":
            img = Image.open(filepath)
            tex = TF.pil_to_tensor(img).float() / 255.0
            if tex.shape[0] == 4:
                tex = tex[:3]
            if tex.ndim == 2:
                tex = tex.unsqueeze(0)
            tex = tex.contiguous()
            image = tex
        case "scikit":
            loaders = {
                "astronaut": skdata.astronaut,
                "camera": skdata.camera,
                "chelsea": skdata.chelsea,
                "coffee": skdata.coffee,
                "horse": skdata.horse,
                "rocket": skdata.rocket,
                "cat": skdata.cat,
                "text": skdata.text,
            }
            if args.scikit_image not in loaders:
                raise ValueError(f"Unknown scikit_image: {args.scikit_image}")
            img = loaders[args.scikit_image]()
            ten = torch.from_numpy(img).float() / 255.0
            ten = ten.unsqueeze(0) if ten.ndim == 2 else ten.permute(2, 0, 1)
            ten = ten[:3].contiguous()
            image = ten
        case "gabor":
            return SteerableImage(impl="gabor", base_orientation=base_orientation)
        case _:
            raise ValueError(f"Unknown im_kind: {kind}")
    image = _preprocess_tensor(kind, image)
    return SteerableImage(impl="rotate", image=image, base_orientation=base_orientation)


def _compute_profiles(x: torch.Tensor) -> dict[str, dict]:
    out = {}
    for method in args.profile_methods:
        angles_rad, power = compute_angular_profile(
            x,
            angular_resolution=args.angular_resolution,
            method=method,
            renormalize=renormalize,
            apply_mask=args.apply_mask,
            mask_kind=args.mask_kind,
            sense=args.angular_profile_sense,
            filtering=args.angular_profile_filtering,
            **_profile_kwargs(method),
        )
        angles_rad = torch.as_tensor(angles_rad)
        power = torch.as_tensor(power)
        if method == "binning":
            angles_rad = torch.remainder(math.pi - angles_rad, math.pi)
        max_idx = torch.argmax(power)
        peak_deg = angles_rad[max_idx].item() * 180.0 / math.pi
        out[method] = {"angles_rad": angles_rad, "power": power, "peak_deg": peak_deg}
    return out


def _estimate_peak_deg(x: torch.Tensor, method: str) -> float:
    angles_rad, power = compute_angular_profile(
        x,
        method=method,
        renormalize=renormalize,
        apply_mask=args.apply_mask,
        mask_kind=args.mask_kind,
        sense=args.angular_profile_sense,
        filtering=args.angular_profile_filtering,
        **_profile_kwargs(method),
    )
    angles_rad = torch.as_tensor(angles_rad)
    power = torch.as_tensor(power)
    if method == "binning":
        angles_rad = torch.remainder(math.pi - angles_rad, math.pi)
    max_idx = torch.argmax(power)
    return angles_rad[max_idx].item() * 180.0 / math.pi


def _make_register_fn(method: str):
    def _register_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        est_a = _estimate_peak_deg(a, method)
        est_b = _estimate_peak_deg(b, method)
        _, rotation_deg = register(
            a,
            b,
            est_a,
            est_b,
            criterion=args.registration_criterion,
            return_rotation=True,
        )
        return torch.tensor(math.radians(rotation_deg), device=a.device, dtype=a.dtype)

    return _register_fn


def _compute_display_masks(height: int, width: int) -> dict[str, torch.Tensor]:
    u = torch.fft.fftfreq(height, d=1.0)
    v = torch.fft.fftfreq(width, d=1.0)
    u, v = torch.meshgrid(u, v, indexing="ij")
    radius = torch.sqrt(u**2 + v**2)
    angle = torch.atan2(u, v)
    orientation = torch.tensor(
        args.display_mask_orientation * math.pi / 180.0,
        dtype=radius.dtype,
        device=radius.device,
    )
    cake = compute_cake_wavelet_filter(
        radius,
        angle,
        orientation=orientation,
        resolution=(height, width),
        sense=args.angular_profile_sense,
        filtering=args.angular_profile_filtering,
    )
    radon = compute_ridge_filter(
        radius,
        angle,
        orientation=orientation,
        resolution=(height, width),
        sense=args.angular_profile_sense,
        filtering=args.angular_profile_filtering,
    )
    return {"Cake wavelet": cake, "Ridge": radon}


def _plot_display_masks(
    masks: dict[str, torch.Tensor], height: int, width: int
) -> None:
    # Center DC for display by fftshifting the mask grids.
    shifted_masks = {label: torch.fft.fftshift(mask) for label, mask in masks.items()}
    values = torch.stack([mask.float() for mask in shifted_masks.values()])
    vmin = values.min().item()
    vmax = values.max().item()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), constrained_layout=True)
    last_im = None
    for ax, (label, mask) in zip(axes, shifted_masks.items(), strict=False):
        last_im = ax.imshow(mask.cpu(), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=list(axes),
            location="right",
            shrink=0.85,
            pad=0.02,
        )
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"angular_masks_{height}x{width}.svg")
    fig.savefig(out_dir / f"angular_masks_{height}x{width}.pdf")
    _OPEN_FIGURES.append(fig)
    _show_figure(fig)


def _show_figure(fig: plt.Figure) -> None:
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    fig.show()
    plt.show(block=False)
    plt.pause(0.01)


im_a = _make_image(
    args.im_kind,
    args.texture_path if args.im_kind == "texture" else args.dicom_file,
    base_orientation=args.base_orientation_a,
)
im_b = _make_image(
    args.im_kind,
    args.texture_path if args.im_kind == "texture" else args.dicom_file,
    base_orientation=args.base_orientation_b,
)

x_a = im_a.steer(0.0)
x_b = im_b.steer(0.0)

renormalize = _renormalize_flag()

display_masks = _compute_display_masks(
    args.display_mask_height, args.display_mask_width
)
_plot_display_masks(display_masks, args.display_mask_height, args.display_mask_width)

# helper to compute registration metric for reporting
if args.registration_metric == "psnr":

    def _reg_metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return peak_signal_noise_ratio(a, b)

else:

    def _reg_metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return structural_similarity_index_measure(
            a.unsqueeze(0), b.unsqueeze(0), data_range=1.0
        )


# Compute angular profiles and peak estimates for both images
profiles_a = _compute_profiles(x_a)
profiles_b = _compute_profiles(x_b)

ref_method = args.profile_methods[0]
est_a = profiles_a[ref_method]["peak_deg"]
est_b = profiles_b[ref_method]["peak_deg"]
# keep delta in [-90, 90] to respect 180° symmetry
aligned_b = register(x_a, x_b, est_a, est_b, criterion=args.registration_criterion)

# Plot images and profiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_a, ax_b, ax_align, ax_prof = axes.flatten()
ax_a.imshow(x_a.permute(1, 2, 0), cmap="gray")
ax_a.set_title(f"Image A @ {args.base_orientation_a:.1f}°")
ax_a.axis("off")
ax_b.imshow(x_b.permute(1, 2, 0), cmap="gray")
ax_b.set_title(f"Image B @ {args.base_orientation_b:.1f}°")
ax_b.axis("off")
ax_align.imshow(aligned_b.permute(1, 2, 0), cmap="gray")
ax_align.set_title("Image B aligned")
ax_align.axis("off")
for method, prof in profiles_a.items():
    angles_deg = prof["angles_rad"] * 180.0 / math.pi
    ax_prof.plot(
        angles_deg, prof["power"], label=f"A {method} (peak {prof['peak_deg']:.1f}°)"
    )
for method, prof in profiles_b.items():
    angles_deg = prof["angles_rad"] * 180.0 / math.pi
    ax_prof.plot(
        angles_deg,
        prof["power"],
        linestyle="--",
        label=f"B {method} (peak {prof['peak_deg']:.1f}°)",
    )
ax_prof.axvline(est_a, color="k", linestyle=":", linewidth=1.0, label="A ref")
ax_prof.axvline(est_b, color="gray", linestyle=":", linewidth=1.0, label="B ref")
ax_prof.set_yscale("log")
ax_prof.set_xlabel("Angle (°)")
ax_prof.set_ylabel("Power")
ax_prof.set_xticks(range(0, 181, 45))
ax_prof.set_title("Angular profiles and peaks")
ax_prof.legend()
fig.tight_layout()
_OPEN_FIGURES.append(fig)
_show_figure(fig)

# Standalone angular profile figure.
method_display = {
    "binning": "Binning",
    "cake_wavelet": "Cake wavelet",
    "ridge": "Ridge",
}
color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
method_colors = {
    method: color_cycle[idx % len(color_cycle)] if color_cycle else None
    for idx, method in enumerate(args.profile_methods)
}
profiles_fig, (profiles_ax_a, profiles_ax_b) = plt.subplots(
    1,
    2,
    figsize=(3.5, 2),
    subplot_kw={"projection": "polar"},
    sharex=False,
    sharey=False,
    constrained_layout=True,
)


def _expand_angles_half_turn(
    angles_rad: torch.Tensor, power: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Normalize to [0, pi) then duplicate to [0, 2*pi).
    base_angles = torch.remainder(angles_rad, math.pi)
    order = torch.argsort(base_angles)
    base_angles = base_angles[order]
    base_power = power[order]
    return (
        torch.cat([base_angles, base_angles + math.pi]),
        torch.cat([base_power, base_power]),
    )


handles = []
labels = []
opacity_model = {
    "binning": 0.35,
    "ridge": 0.9,
    "cake_wavelet": 0.7,
}
for method, prof in profiles_a.items():
    angles_full, power_full = _expand_angles_half_turn(
        prof["angles_rad"], prof["power"]
    )
    power_full = power_full.clamp(min=1e-16)
    (line_a,) = profiles_ax_a.plot(
        angles_full,
        power_full,
        label=method_display.get(method, method),
        color=method_colors.get(method),
        linestyle="-",
        alpha=opacity_model[method]
    )
    handles.append(line_a)
    labels.append(line_a.get_label())
for method, prof in profiles_b.items():
    angles_full, power_full = _expand_angles_half_turn(
        prof["angles_rad"], prof["power"]
    )
    power_full = power_full.clamp(min=1e-16)
    profiles_ax_b.plot(
        angles_full,
        power_full,
        color=method_colors.get(method),
        linestyle="-",
        alpha=opacity_model[method]
    )
all_power = torch.cat(
    [prof["power"].flatten() for prof in profiles_a.values()]
    + [prof["power"].flatten() for prof in profiles_b.values()]
)
radial_min = max(1e-16, (all_power.min().item() or 0.0) / 2.0)
radial_max = (all_power.max().item() or 1.0) * 2.0
if radial_min <= 0.0:
    radial_min = radial_max / 1e6
for ax in (profiles_ax_a, profiles_ax_b):
    ax.set_rscale("log")
    ax.set_ylabel("")
    ax.set_ylim(radial_min, radial_max)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_thetagrids(range(0, 360, 45))
    ax.set_yticks([])
    ax.set_yticklabels([])
legend_cols = max(1, len(args.profile_methods))
legend = profiles_fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=legend_cols,
    frameon=False,
    bbox_transform=profiles_fig.transFigure,
)
profiles_fig.subplots_adjust(top=0.78, wspace=0.25)
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
profiles_fig.savefig(
    out_dir / "angular_profiles.svg",
    bbox_inches="tight",
    bbox_extra_artists=(legend,),
)
profiles_fig.savefig(
    out_dir / "angular_profiles.pdf",
    bbox_inches="tight",
    bbox_extra_artists=(legend,),
)
_OPEN_FIGURES.append(profiles_fig)
_show_figure(profiles_fig)

# Summarize metrics for registration
rows = []
for method in args.profile_methods:
    prof_a = profiles_a[method]
    prof_b = profiles_b[method]
    angles_a = prof_a["angles_rad"]
    angles_b = prof_b["angles_rad"]
    power_a = prof_a["power"]
    power_b = prof_b["power"]
    circ_var_a = circular_variance(angles_a, power_a, dim=0, oriented=False)
    circ_var_b = circular_variance(angles_b, power_b, dim=0, oriented=False)
    method_aligned, method_rotation = register(
        x_a,
        x_b,
        prof_a["peak_deg"],
        prof_b["peak_deg"],
        criterion=args.registration_criterion,
        return_rotation=True,
    )
    metric_val = _reg_metric(x_a, method_aligned).item()
    reg_angle = math.radians(method_rotation)
    true_delta_deg = args.base_orientation_a - args.base_orientation_b
    ang_dist = angular_distance(
        torch.tensor(math.radians(true_delta_deg)), torch.tensor(reg_angle)
    ).item()
    ang_dist = math.degrees(ang_dist)
    equiv_error = equivariance_metric(
        x_a,
        x_b,
        _make_register_fn(method),
        num_angles=args.equivariance_angles,
        seed=args.equivariance_seed,
    ).item()
    equiv_error_deg = math.degrees(equiv_error)
    rows.append(
        {
            "method": method,
            "circular_variance_a": circ_var_a.item(),
            "circular_variance_b": circ_var_b.item(),
            "angular_distance": ang_dist,
            "equivariance_error_deg": equiv_error_deg,
            f"{args.registration_metric}": metric_val,
        }
    )
metrics_df = pd.DataFrame(rows)
print("\nOrientation registration metrics:")
print(metrics_df)
print("\nOrientation registration metrics (JSONL):")
for record in metrics_df.to_dict(orient="records"):
    print(json.dumps(record))

reg_metric_val = _reg_metric(x_a, aligned_b)
print(
    f"{args.registration_metric.upper()} of aligned B vs A: {reg_metric_val.item():.2f}"
)
print(f"\nReference method: {ref_method}")
print(f"Estimated A: {est_a:.2f}°, Estimated B: {est_b:.2f}°")

methods = ("binning", "cake_wavelet", "ridge")


def _hold_plot_windows() -> None:
    if not args.hold_plots:
        return
    # Keep GUI event loop alive until all figures are closed.
    while plt.get_fignums():
        plt.pause(0.1)


_hold_plot_windows()
