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

# +
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import os

# %matplotlib inline
# -

ds_num = input("Dataset: #")
exp_dir = input("Experiment Directory: ")

if os.path.exists(exp_dir + "/artifacts/overlay"):
    print("Directory already exists")
    pass
else:
    os.mkdir(exp_dir + "/artifacts/overlay")

df = pd.read_csv(exp_dir + "/artifacts/pred_{:03d}_copy.csv".format(int(ds_num)))


def get_info(df, line):
    value = []
    params = ["e_trans3d", "e_trans2d", "e_orient", "e_phi", "e_gamma"]
    for param in params:
        value.append(str(round(df[param][line], 2)))

    return value


for i in tqdm.trange(100):
    orig = cv2.imread("../../Database/ds_021/val/img_{:05d}.jpg".format(i + 1))
    pred = cv2.imread(exp_dir + "/artifacts/pred_imgs/img_{:05d}.png".format(i + 1))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGBA)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGRA2RGBA)
    #     pred[:, :, 0:2] = 0
    blended = cv2.addWeighted(src1=orig, alpha=0.8, src2=pred, beta=0.4, gamma=0)
    im = cv2.cvtColor(blended, cv2.COLOR_RGBA2BGR)
    info = get_info(df, i)

    txt0 = "Error (3D):     {} mm".format(info[0])
    txt1 = "Error (2D):     {} px".format(info[1])
    txt2 = "Error (Orient): {} deg".format(info[2])
    txt3 = "Error (Joint):  {} deg".format(info[3])
    txt4 = "Error (Roll):   {} deg".format(info[4])

    im = cv2.putText(
        im,
        txt0,
        (50, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (
            255,
            255,
            0,
        ),
        2,
    )
    im = cv2.putText(
        im,
        txt1,
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (
            255,
            255,
            0,
        ),
        2,
    )
    im = cv2.putText(
        im,
        txt2,
        (50, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (
            255,
            255,
            0,
        ),
        2,
    )
    im = cv2.putText(
        im,
        txt3,
        (50, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (
            255,
            255,
            0,
        ),
        2,
    )
    im = cv2.putText(
        im,
        txt4,
        (50, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (
            255,
            255,
            0,
        ),
        2,
    )
    cv2.imwrite(exp_dir + "/artifacts/overlay/img_{:05d}.png".format(i + 1), im)


