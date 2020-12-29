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



import bpy
import pandas as pd
import numpy as np
from random import randrange

unit2mm = 1 / 1000
deg2rad = np.pi / 180

dataset_num = 23
file_format = 'jpg'   # jpg/png

csv_dir = 'C:/Users/admin/Google Drive/deepest3d/Database/ds_{:03d}'.format(dataset_num)
file_path_train = csv_dir + '/train_{:03d}.csv'.format(dataset_num)
file_path_val = csv_dir + '/val_{:03d}.csv'.format(dataset_num)

df_train = pd.read_csv(file_path_train, dtype='float32')
df_val = pd.read_csv(file_path_val, dtype='float32')

df_dict = {'train': df_train, 'val': df_val}

forceps = bpy.data.objects['forceps']
free_edge = bpy.data.objects['Edge_free']
f_loc = forceps.location
f_rot = forceps.rotation_euler
fe_rot = free_edge.rotation_euler


def make_img():
    for phase in ('train', 'val'):
        for i in range(len(df_dict[phase])):

            # Translation
            f_loc.x, f_loc.y, f_loc.z = df_dict[phase].loc[i, 'x':'z'] * unit2mm

            # Rotation (Euler)
            f_rot.z, f_rot.x, f_rot.y = df_dict[phase].loc[i, 'gamma':'beta'] * deg2rad

            fe_rot.x = - (df_dict[phase].loc[i, 'phi'] + 90) * deg2rad

            bpy.ops.render.render()
            bpy.data.images['Render Result'].save_render(
                filepath=csv_dir + '/{}/img_{:05d}.{}'.format(phase, i + 1, file_format))


make_img()
