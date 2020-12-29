import bpy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

forceps = bpy.data.objects['forceps']
free_edge = bpy.data.objects['Edge_free']
f_loc = forceps.location
f_rot = forceps.rotation_euler
fe_rot = free_edge.rotation_euler

ds_dir = '../../Database/JIGSAWS'
operation = {'kt': 'Knot_Tying', 'np': 'Needle_Passing', 'su': 'Suturing'}
df = pd.read_csv(ds_dir + '/' + operation['k'] + '/kinematics/AllGestures/' + operation['k'] + '_B001.csv')

r_loc = df.loc[:, ['MR_x', 'MR_y', 'MR_z']]
r_matrix = df.loc[:, ['MR_r{}'.format(i) for i in range(9)]]
r_grip = df['MR_angle']

for i in range(1):
    f_loc.x, f_loc.y, f_z = r_loc.loc[i] / 10
    f_loc.z = - f_z
    r_rot = Rotation.from_matrix(np.array(r_matrix.loc[i]).reshape(3, 3))
    f_rot.z, f_rot.x, f_rot.y = r_rot.as_euler('zxy')
    fe_rot.x = r_grip.loc[i]
