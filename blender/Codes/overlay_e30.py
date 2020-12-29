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

df = pd.read_csv(exp_dir + "/artifacts/roll_pred_{:03d}.csv".format(int(ds_num)))
e30_df = pd.read_csv(exp_dir + "/artifacts/e30.csv", header=None)


def get_info(df, line):
    return str(round(df["e_gamma"][line], 2))


for i in e30_df[0]:
    i = int(i)
    print(i)
    orig = cv2.imread("../../Database/ds_021/val/img_{:05d}.jpg".format(i))
    pred = cv2.imread(exp_dir + "/artifacts/e30/img_{:05d}.png".format(i))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGBA)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGRA2RGBA)
    #     pred[:, :, 0:2] = 0
    #     blended = cv2.addWeighted(src1=orig, alpha=0.8, src2=pred, beta=0.4, gamma=0)
    im_cat = cv2.vconcat([orig, pred])
    im224 = cv2.resize(im_cat, (224, 448))
    im = cv2.cvtColor(im224, cv2.COLOR_RGBA2BGR)
    info = get_info(df, i - 1)

    # txt = 'Error (Roll):   {} deg'.format(info)
    #
    # im = cv2.putText(im, txt, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0, ), 2)
    cv2.imwrite(exp_dir + "/artifacts/overlay_e30_224/img_{:05d}.png".format(i), im)

print("Finished")
