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

# Generating evaluation dataset
import asyncio

import numpy as np
from sksurgerynditracker.nditracker import NDITracker
import cv2
from scipy.spatial.transform import Rotation
import pandas as pd
from IPython.display import clear_output

# -

ROMFILE_DIR = "./tracking/Slicer/rom_config/"

rom_list = ["8700339", "custom_001"]

TRACKER_SETTINGS = {
    "tracker type": "polaris",
    "romfiles": list(map(lambda x: ROMFILE_DIR + x + ".rom", rom_list)),
}


class SyncCamTrak:
    def __init__(self):
        print("\nInitializing tracker...")
        self.tracker = NDITracker(TRACKER_SETTINGS)
        self.tracker.start_tracking()

        print("Initializing camera...\n")
        self.cap = cv2.VideoCapture(0)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video = cv2.VideoWriter('eval.mp4', fourcc, fps, (w, h))

    def get_transform(self, tool, mat):
        assert mat.shape == (4, 4)
        x = mat[0, 3]
        y = mat[1, 3]
        z = mat[2, 3]
        rot_mat = mat[:3, :3]
        rot = Rotation.from_matrix(rot_mat).as_euler()
        alpha, beta, gamma = np.rad2deg(rot)
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = rot_mat.flatten()
        print('''
        =================
        ID: {}
        -----------------
        Tx: {:.2f}
        Ty: {:.2f}
        Tz: {:.2f}
        Rx: {:.2f} deg
        Ry: {:.2f} deg
        Rz: {:.2f} deg
        =================
        '''.format(tool, x, y, z, alpha, beta, gamma))
        return x, y, z, r1, r2, r3, r4, r5, r6, r7, r8, r9

    def on(self):
        k = 0
        tool_dict = {'1': 'forceps', '2': 'endoscope'}
        transform_dict = {'forceps': [], 'endoscope': []}
        while True:
            clear_output(wait=True)
            ret, frame = self.cap.read()
            cv2.imshow('camera', frame)
            self.video.write(frame)
            id_, _, _, mat, _ = self.tracker.get_frame()
            for i in id_:
                tool = tool_dict[str(i)]
                transform = self.get_transform(tool, mat[i-1])
                transform_dict[tool].append(transform)


            k += 1
            if k == 100:
#             if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for tool in ['forceps', 'endoscope']:
            df = pd.DataFrame(transform_dict[tool], columns=['x', 'y', 'z', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'])
            df.to_csv('{}.csv'.format(tool))
        print('\nFile save successed!!')
        self.cap.release()
        cv2.destroyAllWindows()

main = SyncCamTrak()

input(
      '\n==============================\n'
      'Press ENTER to start recording\n'
      '==============================\n')

if __name__ == '__main__':
    main.on()
