# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Import libraries
import os
import argparse
import yaml
import numpy as np
from random import uniform, random
import pandas as pd
from scipy.spatial.transform import Rotation
import seaborn
import matplotlib.pyplot as plt
import importlib


os.chdir("../..")

rep = importlib.import_module("utils.reprojection")

# Args
args = argparse.ArgumentParser()
args.add_argument(
    "-train",
    "--train-sample-num",
    default=10000,
    type=int,
    help="the number of training data",
)
args.add_argument(
    "-val",
    "--val-sample-num",
    default=2000,
    type=int,
    help="the number of validation data",
)
args.add_argument("--x-min", default=13, help="x min [mm]")
args.add_argument("--y-min", default=6.5, help="y min [mm]")
args.add_argument("--z-min", default=40, help="z min [mm]")
args.add_argument("--z-max", default=100, help="z max [mm]")
args.add_argument("--alpha-min", default=-60, help="alpha min [deg]")
args.add_argument("--alpha-max", default=60, help="alpha max [deg]")
args.add_argument("--beta-min", default=-60, help="beta min [deg]")
args.add_argument("--beta-max", default=60, help="beta max [deg]")
args.add_argument("--gamma-min", default=-179, help="gamma min [deg]")
args.add_argument("--gamma-max", default=179, help="gamma max [deg]")
args.add_argument("--phi-max", default=45, help="max opening angle of the edge [deg]")


conf = args.parse_args(args=[])


# ### Origin-Line Distance (3D)
# Position: $\ \vec{r} = (\alpha, \beta, \gamma)$\
# Direction vector: $\ \vec{n} = (n_x, n_y, n_z)$\
# Line: $\ \vec{p}(t) = \vec{r} + t\vec{n}$\
# $$
# \vec{p}(h)\cdot\vec{n} = 0 \\
# h = - \frac{\vec{r}\cdot\vec{n}}{\|\vec{n}\|^2} \\
# d = \|\vec{r}-h\vec{n}\|_2
# $$

def o2line(r, n):
    s = 0
    for i in range(3):
        s += (r[i] - np.dot(r, n) / ((np.linalg.norm(n, ord=2)) ** 2) * n[i]) ** 2
    return np.sqrt(s)


# ### Make a dataset directory

ds_num = 1  # Initialize ds_num
while True:
    ds_dir = "Database/ds_{:03d}".format(ds_num)
    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)
        os.makedirs(ds_dir + "/train")
        os.makedirs(ds_dir + "/val")
        break

    else:
        ds_num += 1
        continue

sample_num_dict = {"train": conf.train_sample_num, "val": conf.val_sample_num}

# ### Parameters
# $$
# \theta = \{x, y, z, x_{2d}, y_{2d}, n_x, n_y, n_z, \phi, \gamma, \alpha, \beta\}
# $$

# +
phase_dict = {"train": "training", "val": "validation"}


def make_csv():
    for phase in ["train", "val"]:
        data = []
        i = 0
        while i < sample_num_dict[phase]:
            # Position
            z = uniform(conf.z_min, conf.z_max)
            x_range = conf.x_min * (z / conf.z_min)
            y_range = conf.y_min * (z / conf.z_min)
            x = uniform(-x_range, x_range)
            y = uniform(-y_range, y_range)
            r = np.array([x, y, z])

            # 2D position
            x_2d, y_2d = rep.repon2plane(r, a_ratio=(16, 9), fov=70)

            alpha = uniform(
                conf.alpha_min, conf.alpha_max
            )  # Rotation around local x-axis
            beta = uniform(conf.beta_min, conf.beta_max)  # Rotation around local y-axis
            gamma = uniform(
                conf.gamma_min, conf.gamma_max
            )  # Rotation around local z-axis

            rot_matrix = Rotation.from_euler("zxy", [gamma, alpha, beta], degrees=True)

            n0 = np.array([0.0, 0.0, -1.0])  # Initial vector
            n = rot_matrix.apply(n0)  # Direction vector

            phi = uniform(0, conf.phi_max)

            if n[2] >= -0.5 or o2line(r, n) < 5:
                continue
            else:
                row = [x, y, -z, n[0], n[1], n[2], phi, gamma, alpha, beta, x_2d, y_2d]
                data.append(row)
                i += 1

        df = pd.DataFrame(
            data=data,
            columns=[
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "phi",
                "gamma",
                "alpha",
                "beta",
                "x_2d",
                "y_2d",
            ],
        )

        alpha_list = [row[8] for row in data]
        beta_list = [row[9] for row in data]
        gamma_list = [row[7] for row in data]
        nx_list = [row[3] for row in data]
        ny_list = [row[4] for row in data]
        nz_list = [row[5] for row in data]

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        ax1.hist(
            [alpha_list, beta_list, gamma_list],
            bins=36,
            range=(-180, 180),
            label=[r"$\alpha$", r"$\beta$", r"$\gamma$"],
        )
        ax1.legend()
        ax1.set_title(phase_dict[phase].capitalize())
        ax1.set_xlabel("Degrees")
        ax1.set_ylabel("Frequency")
        ax2.hist(
            [nx_list, ny_list, nz_list],
            bins=40,
            range=(-1, 1),
            label=[r"$n_x$", r"$n_y$", r"$n_z$"],
        )
        ax2.legend()
        ax2.set_title(phase_dict[phase].capitalize())
        ax2.set_xlabel(r"$[a.u.]$")
        ax2.set_ylabel("Frequency")
        fig.savefig(ds_dir + "/{}_data_distribution.png".format(phase), dpi=300)

        # save as csv
        df.to_csv(ds_dir + "/{}_{:03d}.csv".format(phase, ds_num), index=False)

        yml = {
            "pose": {"euler": {"axis": "zxy", "rotation": "intrinsic"}},
            "translation": {
                "x_range": 65,
                "y_range": 35,
                "z_min": conf.z_min,
                "z_max": conf.z_max,
            },
            "articulation": {"phi_max": conf.phi_max},
            "data": {"train": conf.train_sample_num, "val": conf.val_sample_num},
            "params": 12,
            "o2line": 5,
        }

        with open(ds_dir + "/ds_config.yaml", "w") as file:
            yaml.dump(yml, file, default_flow_style=False)


# -

make_csv()

print(
    "\n=========================\n"
    "PROCESS COMPLETED    \n"
    "=========================\n"
)
