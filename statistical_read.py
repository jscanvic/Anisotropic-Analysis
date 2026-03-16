import json

import numpy as np


def main():
    data = None
    with open("valid_statistical.json") as json_file:
        data = json.load(json_file)

    if data is None:
        raise RuntimeError(
            "After trying to load valid_statistical.json, data is None ..."
        )

    [
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
    ] = data

    n0 = ["Cake wavelet", "Ridge", "Binning"]
    names = []
    for k, n in enumerate(n0):
        names.append(n + " pdf_dist")
        names.append(n + " circ_var")
        names.append(n + " l2_reg")
        names.append(n + " rot_eq")
        names.append(n + " vec_ang")

    r = int(len(data) / 3)

    for i in range(3):
        n = n0[i]

        line = f"{n}"

        id_vec_ang = i*r + 4
        mean_k = np.mean(data[id_vec_ang])
        disp = round(mean_k, 2)
        line += f" & {disp}"

        id_rot_eq = i*r + 3
        db = -10 * np.log10(data[id_rot_eq])
        mean_k = np.mean(db)
        disp = round(mean_k, 2)
        line += f" & {disp}"

        print(line + "\\\\")

    print("--")
    for i in range(3):
        n = n0[i]

        line = f"{n}"

        id_vec_ang = i*r + 4
        std_k = np.std(data[id_vec_ang])
        disp = round(std_k, 2)
        line += f" & {disp}"

        id_rot_eq = i*r + 3
        db = -10 * np.log10(data[id_rot_eq])
        std_k = np.std(db)
        disp = round(std_k, 2)
        line += f" & {disp}"

        print(line + "\\\\")


if __name__ == "__main__":
    main()
