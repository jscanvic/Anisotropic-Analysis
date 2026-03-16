import pydicom.pixel_data_handlers
import torch


def load_scan(filepath: str) -> torch.Tensor:
    r"""
    Load a CT scan from a DICOM file

    :param str filepath: The path of the DICOM file
    :return: torch.Tensor, the CT scan (1, H, W), linear attenuation coefficients (LAC) in cm^-1
    """
    if filepath is None:
        raise ValueError("dicom_path must be specified for impl='rotate'")

    ds = pydicom.dcmread(filepath)
    x = ds.pixel_array
    x = pydicom.pixel_data_handlers.apply_rescale(x, ds)
    x = x.clip(min=-1000)  # Clip to minimum of -1000 HU
    x = torch.from_numpy(x)

    # to linear attenuation coefficients (LAC) in cm^-1
    # LAC of water: 0.0958 cm^-1
    # LAC of air: 0.0001 cm^-1
    # HU = 1000 * (LAC - LAC_water) / (LAC_water - LAC_air)
    hu_to_lac_mode = "simple"
    if hu_to_lac_mode == "realistic":
        x = (0.0958 - 0.0001) / 1000 * x + 0.0958
    elif hu_to_lac_mode == "simple":
        x = (
            x / 1000 + 1
        )  # Assume a LAC for water of 1 cm^-1 and a LAC for air of 0 cm^-1
    else:
        raise ValueError(f"Unknown hu_to_lac_mode: {hu_to_lac_mode}")

    return x.unsqueeze(0)
