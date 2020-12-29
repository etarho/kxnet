"""Generating a sim video"""

import bpy
import os
import pandas as pd
import numpy as np
from random import randrange

unit2mm = 1 / 1000

dataset_num = 21
file_format = 'png'   # jpg/png

csv_dir = r''
file_path = csv_dir + '/pred_{:03d}.csv'.format(dataset_num)
if os.path.exists(csv_dir + '/pred_imgs'):
    pass
else:
    os.mkdir(csv_dir+'/pred_imgs')

df = pd.read_csv(file_path, dtype='float32')

forceps = bpy.data.objects['forceps']
free_edge = bpy.data.objects['Edge_free']
f_loc = forceps.location
f_rot = forceps.rotation_euler
fe_rot = free_edge.rotation_euler

bpy.context.scene.render.film_transparent = True

def make_img():
    for i in range(100):

        # Translation
        f_loc.x, f_loc.y, z = df.iloc[i][['x', 'y', 'z']] * unit2mm
        f_loc.z = -z

        # Rotation (Euler)
        nx, ny = df.iloc[i][['nx', 'ny']]
        f_rot.x = np.arcsin(ny)
        f_rot.y = np.arccos(nx/np.sqrt(1-ny**2)) - np.pi/2
        f_rot.z = np.deg2rad(df['gamma'][i])

        fe_rot.x = - np.deg2rad(df['phi'][i] + 90)

        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(
            filepath=csv_dir + '/pred_imgs/img_{:05d}.{}'.format(i + 1, file_format))


make_img()
