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

"""Generating a sim video"""

from random import randrange
import bpy
import numpy as np
import pandas as pd

unit2mm = 1 / 1000
deg2rad = np.pi / 180

dataset_num = 23
file_format = "jpg"  # jpg/png

csv_dir = "C:/Users/admin/Google Drive/deepest3d/Database/ds_{:03d}".format(
    dataset_num)
file_path_train = csv_dir + "/train_{:03d}.csv".format(dataset_num)
file_path_val = csv_dir + "/val_{:03d}.csv".format(dataset_num)

df_train = pd.read_csv(file_path_train, dtype="float32")
df_val = pd.read_csv(file_path_val, dtype="float32")

df_dict = {"train": df_train, "val": df_val}

forceps = bpy.data.objects["forceps"]
free_edge = bpy.data.objects["Edge_free"]
f_loc = forceps.location
f_rot = forceps.rotation_euler
fe_rot = free_edge.rotation_euler


def make_img():
    for phase in ("train", "val"):
        for i in range(len(df_dict[phase])):

            # Translation
            f_loc.x, f_loc.y, f_loc.z = df_dict[phase].loc[i, "x":"z"] * unit2mm

            # Rotation (Euler)
            f_rot.z, f_rot.x, f_rot.y = df_dict[phase].loc[
                i, "gamma":"beta"] * deg2rad

            fe_rot.x = -(df_dict[phase].loc[i, "phi"] + 90) * deg2rad

            # Import and transform a background image
            bg_img_path = "C:/Users/admin/Google Drive/deepest3d/Database/surgical_videos/movie_002_L/movie_002_L_001"
            img_num = randrange(174)
            bpy.ops.import_image.to_plane(
                shader="SHADELESS",
                files=[{
                    "name": bg_img_path + "{:05d}.png".format(img_num)
                }],
            )
            bg = bpy.data.objects["movie_002_L_001" + "{:05d}".format(img_num)]
            bg.location = (0, 0, -0.5)
            # bpy.context.scene.objects.active = bg
            bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))

            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(
                filepath=csv_dir +
                "/{}/img_{:05d}.{}".format(phase, i + 1, file_format))

            objs = bpy.data.objects
            objs.remove(bg, do_unlink=True)


make_img()
